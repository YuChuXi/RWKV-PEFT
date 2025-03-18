import joblib
import torch
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel
import tqdm


class ViTFeatureConfig:
    """ViT特征层配置器"""

    def __init__(
        self,
        model_name="google/vit-large-patch16-384",
        layer_groups={
            "coarse": [-1],  # 最后一层
            "medium": [-4, -3],  # 倒数第4、3层
            "fine": [-6, -5],  # 倒数第6、5层
        },
        tile_size=384,
        max_tiles=4,
        batch_size=16,
    ):
        """
        :param tile_size: 图像分块尺寸
        :param max_tiles: 最大分块数（宽高各分块数）
        """
        self.model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        self.processor = ViTImageProcessor.from_pretrained(
            model_name,
            image_mean=[0.5],  # 单通道均值
            image_std=[0.5]    # 单通道标准差
        )
        self.layer_groups = layer_groups
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        self.batch_size = batch_size

        # 注册钩子
        self.feature_maps = {}
        self._register_hooks()

    def _register_hooks(self):
        """动态注册指定层的特征钩子"""
        all_layers = set()
        for group in self.layer_groups.values():
            all_layers.update([self._resolve_layer_idx(idx) for idx in group])

        def hook_generator(layer_idx):
            def hook(module, inputs, outputs):
                self.feature_maps[layer_idx] = outputs.last_hidden_state

            return hook

        for layer_idx in all_layers:
            self.model.encoder.layer[layer_idx].register_forward_hook(
                hook_generator(layer_idx)
            )

    def _resolve_layer_idx(self, index):
        """处理负数索引"""
        if index < 0:
            return self.model.config.num_hidden_layers + index
        return index


class ImageTiler:
    """大尺寸图像分块处理器"""

    def __init__(self, tile_size=384, max_tiles=4):
        self.tile_size = tile_size
        self.max_tiles = max_tiles

    def __call__(self, img):
        """返回分块列表和布局信息"""
        w, h = img.size
        num_tiles_w = min(self.max_tiles, w // self.tile_size)
        num_tiles_h = min(self.max_tiles, h // self.tile_size)

        tiles = []
        positions = []
        for i in range(num_tiles_w):
            for j in range(num_tiles_h):
                left = i * w // num_tiles_w
                upper = j * h // num_tiles_h
                right = (i + 1) * w // num_tiles_w
                lower = (j + 1) * h // num_tiles_h

                tile = img.crop((left, upper, right, lower))
                tiles.append(tile)
                positions.append((i, j))

        return {
            "tiles": tiles,
            "layout": (num_tiles_w, num_tiles_h),
            "original_size": (w, h),
        }


class FeatureCacheManager:
    """特征缓存管理器"""

    def __init__(self, cache_dir="data/features_cache", batch_size=32):
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.cache_dir.mkdir(exist_ok=True)

    def save_features(self, image_id, features):
        """保存特征到.pth文件"""
        # 自动转换设备并保存完整元数据
        save_dict = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        torch.save(save_dict, self._get_pth_path(image_id))

    def load_features(self, image_id, device="cpu"):
        """从.pth文件加载特征"""
        path = self._get_pth_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"特征文件 {path} 不存在")

        # 支持跨设备加载
        return torch.load(path, map_location=torch.device(device))


class ViTFeatureExtractor:
    """批量特征提取流水线"""

    def __init__(self, config):
        self.config = config
        self.tiler = ImageTiler(config.tile_size, config.max_tiles)
        self.cache = FeatureCacheManager()

    @torch.no_grad()
    def extract_batch(self, image_paths):
        """批量处理图像"""
        dataset = ImageDataset(image_paths, self.tiler)
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, collate_fn=collate_tiled_images
        )

        for batch in tqdm.tqdm(loader):
            inputs = self.config.processor(
                images=batch["tiles"], return_tensors="pt"
            ).to(self.config.model.device)

            outputs = self.config.model(**inputs)
            features = self._aggregate_features(batch, outputs)

            for img_id, feat in zip(batch["image_ids"], features):
                self.cache.save_features(img_id, feat)

    def _aggregate_features(self, batch, outputs):
        """聚合分块特征"""
        batch_features = []
        ptr = 0

        for layout in batch["layouts"]:
            num_tiles = layout[0] * layout[1]
            tile_feats = {}

            # 收集当前图像所有分块的特征
            for key in self.config.layer_groups:
                layer_feats = []
                for layer_idx in self.config.layer_groups[key]:
                    feat = self.config.feature_maps[layer_idx][ptr : ptr + num_tiles]
                    layer_feats.append(feat.mean(dim=0))  # 分块平均

                tile_feats[key] = torch.stack(layer_feats).mean(dim=0)

            ptr += num_tiles
            batch_features.append(tile_feats)

        return batch_features


class TensorInjector:
    """张量注入器"""

    def __init__(self, cache_dir):
        self.cache = FeatureCacheManager(cache_dir)

    def inject_features(self, x, image_ids, positions):
        """
        :param x: 目标张量 (B, T, C)
        :param image_ids: 对应样本的特征ID列表
        :param positions: 注入位置列表[(b,t), ...]
        :return: 修改后的张量
        """
        modified = x.clone()

        for (b, t), img_id in zip(positions, image_ids):
            features = self.cache.load_features(img_id)
            n = len(features)

            if t + n > x.shape[1]:
                raise ValueError(f"样本{b}位置{t}+{n}超出范围")

            # 按层级顺序注入
            ordered_feats = [features["coarse"], features["medium"], features["fine"]][
                :n
            ]

            for i, feat in enumerate(ordered_feats):
                modified[b, t + i] = feat.to(x.device)

        return modified


# 辅助类和函数
class ImageDataset(Dataset):
    def __init__(self, image_paths, tiler):
        self.image_paths = image_paths
        self.tiler = tiler

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        tiled = self.tiler(img)
        return {
            "image_id": Path(self.image_paths[idx]).stem,
            "tiles": tiled["tiles"],
            "layout": tiled["layout"],
        }


def collate_tiled_images(batch):
    """将分块数据整理为批次"""
    return {
        "image_ids": [item["image_id"] for item in batch],
        "tiles": [tile for item in batch for tile in item["tiles"]],
        "layouts": [item["layout"] for item in batch],
    }


def imageid_to_path(img_id):
    path = "/media/yuchuxi/ST1/image_cache/"
    if img_id.find(".") > 4:
        path += img_id[0:4] + "/"
        file = path + img_id[4:]
    else:
        file = path + img_id
    return file


# 使用示例
if __name__ == "__main__":
    # 初始化配置
    config = ViTFeatureConfig(
        model_name="google/vit-large-patch16-384",
        layer_groups={"coarse": [-1], "medium": [-3, -4], "fine": [-5, -6]},
        tile_size=384,
        max_tiles=4,
        batch_size=16,
    )

    # 特征提取流水线
    extractor = ViTFeatureExtractor(config)
    if os.path.isfile("data/image_path_cache"):
        image_paths = joblib.load("data/image_path_cache")
    else:
        image_paths = [
            file
            for file in map(
                imageid_to_path, tqdm.tqdm(joblib.load("data/image_stat").keys())
            )
            if os.path.isfile(file)
        ]
        joblib.dump(image_paths, "data/image_path_cache")
    extractor.extract_batch(image_paths)

    # # 张量注入
    # target_tensor = torch.randn(2, 10, 1024)  # 假设C=1024
    # injector = TensorInjector("features_cache")

    # modified = injector.inject_features(
    #     x=target_tensor,
    #     image_ids=["img1", "img2"],  # 对应文件名stem
    #     positions=[(0, 2), (1, 5)]  # 注入位置
    # )
