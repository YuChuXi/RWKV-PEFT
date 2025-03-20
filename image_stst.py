import joblib
import os
import h5py
import numpy as np
import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import warnings
import logging

# 配置日志
logging.basicConfig(filename="clip_processing.log", level=logging.ERROR)


def imageid_to_file(img_id):
    path = "/media/yuchuxi/ST1/image_cache/"
    if img_id.find(".") > 4:
        path += img_id[0:4] + "/"
        return path + img_id[4:]
    else:
        return path + img_id


class CLIPFeatureExtractor:
    def __init__(
        self,
        model_name: str = "ViT-B/16",
        batch_size: int = 256,
        target_size: int = 224,
        output_path: str = "data/features",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):

        # 硬件配置
        self.device = device
        self.batch_size = batch_size

        # 初始化模型
        self.model, self.preprocess = self._init_clip_model(model_name, target_size)

        # HDF5配置
        self.output_path = output_path
        self.dataset = None
        self.image_idx_set = {}

    def _init_clip_model(self, model_name, target_size):
        """自定义预处理流程，严格控制插值方式"""
        model, preprocess = clip.load(model_name, device=self.device)

        # 重写预处理管道：使用最近邻插值避免引入新数值
        custom_preprocess = Compose(
            [
                Resize(target_size, interpolation=Image.NEAREST),  # 关键：禁用上采样
                CenterCrop(target_size),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        # 初始化存储class embedding的容器
        self.class_embeddings = []

        # 为每个transformer层注册hook
        def hook_fn(module, input, output):
            # 提取class token输出 [batch_size, n_embd]
            cls_emb = output[0, ...]  # 假设class token在位置0
            self.class_embeddings.append(cls_emb.detach())

        # 获取transformer层（假设ViT模型结构为visual.transformer.resblocks）
        for layer in model.visual.transformer.resblocks:
            layer.register_forward_hook(hook_fn)

        return model, preprocess  # custom_preprocess

    def _init_h5_dataset(self, total_samples, n_layer, feature_dim):
        """初始化HDF5数据集，启用压缩和分块存储"""
        self.h5_file = h5py.File(self.output_path + ".h5", "w")
        self.dataset = self.h5_file.create_dataset(
            "features",
            shape=(total_samples, n_layer, feature_dim),
            dtype=np.float32,
            chunks=(self.batch_size, n_layer, feature_dim),
            compression="gzip",
            compression_opts=9,
        )

    def _process_batch(self, image_batch):
        """批量处理逻辑，最大化GPU利用率"""
        with torch.no_grad():
            # 清空历史embedding缓存
            self.class_embeddings = []
            preprocessed = torch.stack(
                [
                    self.preprocess(img).to(self.device)
                    for img in image_batch
                    if img is not None
                ]
            )

            # 前向传播获取特征并触发hook
            features = self.model.encode_image(preprocessed)
            layer_embeddings = torch.stack(self.class_embeddings, dim=1)
            return layer_embeddings.cpu().numpy().astype(np.float32)

    def process_paths(self, image_ids):
        """主处理流程"""

        # 预计算特征维度
        dummy_img = Image.new("RGB", (224, 224))
        _, n_layer, feature_dim = self._process_batch([dummy_img]).shape

        # 初始化HDF5
        self._init_h5_dataset(len(image_ids), n_layer, feature_dim)
        valid_ids = []
        batch_imgs = []
        idx_file = open(self.output_path + ".ids", "w")
        for image_id in tqdm(image_ids):
            try:
                with warnings.catch_warnings(), Image.open(
                    imageid_to_file(image_id)
                ).convert("RGB") as img:
                    warnings.simplefilter("error")
                    batch_imgs.append(img.copy())
                    valid_ids.append(image_id)
                    idx_file.write(image_id+"\n")
            except Exception as e:
                logging.error(f"Error processing {image_id}: {str(e)}")
            if len(batch_imgs) == self.batch_size:
                h5_indices = [
                    len(valid_ids) + idx for idx in range(-self.batch_size, 0)
                ]
                self.dataset[h5_indices] = self._process_batch(batch_imgs)
                self.dataset.flush()
                idx_file.flush()
                batch_imgs = []
        
        # joblib.dump(valid_ids, self.output_path + ".ids")


if __name__ == "__main__":
    if os.path.isfile("data/image_id_cache"):
        image_ids = joblib.load("data/image_id_cache")
    else:
        image_ids = [
            image_id[0]
            for image_id in tqdm(
                sorted(
                    joblib.load("data/image_stat").items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            if os.path.isfile(imageid_to_file(image_id[0]))
        ]
        joblib.dump(image_ids, "data/image_id_cache")

    extractor = CLIPFeatureExtractor(
        batch_size=512,  # 根据GPU显存调整
        target_size=224,  # CLIP标准输入尺寸
        output_path="data/features",
    )

    extractor.process_paths(image_ids)

    # # 张量注入
    # target_tensor = torch.randn(2, 10, 1024)  # 假设C=1024
    # injector = TensorInjector("features_cache")

    # modified = injector.inject_features(
    #     x=target_tensor,
    #     image_ids=["img1", "img2"],  # 对应文件名stem
    #     positions=[(0, 2), (1, 5)]  # 注入位置
    # )
