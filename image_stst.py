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
logging.basicConfig(filename='clip_processing.log', level=logging.ERROR)

class CLIPFeatureExtractor:
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 batch_size: int = 256,
                 target_size: int = 224,
                 h5_path: str = "data/features.h5",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        # 硬件配置
        self.device = device
        self.batch_size = batch_size
        
        # 初始化模型
        self.model, self.preprocess = self._init_clip_model(model_name, target_size)
        
        # HDF5配置
        self.h5_path = h5_path
        self.dataset = None
        self.current_idx = 0

    def _init_clip_model(self, model_name, target_size):
        """自定义预处理流程，严格控制插值方式"""
        model, preprocess = clip.load(model_name, device=self.device)
        
        # 重写预处理管道：使用最近邻插值避免引入新数值
        custom_preprocess = Compose([
            Resize(target_size, interpolation=Image.NEAREST),  # 关键：禁用上采样
            CenterCrop(target_size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                     (0.26862954, 0.26130258, 0.27577711))
        ])
        
        return model, custom_preprocess

    def _init_h5_dataset(self, total_samples, feature_dim):
        """初始化HDF5数据集，启用压缩和分块存储"""
        with h5py.File(self.h5_path, 'w') as hf:
            self.dataset = hf.create_dataset(
                "features",
                shape=(total_samples, feature_dim),
                dtype=np.float32,
                chunks=(self.batch_size, feature_dim),
                compression="gzip",
                compression_opts=9
            )

    def _process_batch(self, image_batch):
        """批量处理逻辑，最大化GPU利用率"""
        with torch.no_grad():
            preprocessed = torch.stack([
                self.preprocess(img).to(self.device) 
                for img in image_batch if img is not None
            ])
            features = self.model.encode_image(preprocessed)
            return features.cpu().numpy().astype(np.float32)

    def process_paths(self, path_list):
        """主处理流程"""
        # 预计算特征维度
        dummy_img = Image.new('RGB', (224, 224))
        feature_dim = self.model.encode_image(
            self.preprocess(dummy_img).unsqueeze(0).to(self.device)
        ).shape[1]
        
        # 初始化HDF5
        self._init_h5_dataset(len(path_list), feature_dim)
        
        # 分批处理
        for i in tqdm(range(0, len(path_list), self.batch_size)):
            batch_paths = path_list[i:i+self.batch_size]
            image_batch = []
            valid_indices = []
            
            # 加载并验证图像
            for idx, path in enumerate(batch_paths):
                try:
                    with warnings.catch_warnings(), Image.open(path) as img:
                        warnings.simplefilter("error")
                        image_batch.append(img.copy())
                        valid_indices.append(idx)
                except Exception as e:
                    logging.error(f"Error processing {path}: {str(e)}")
                    image_batch.append(None)  # 占位
            
            # 过滤无效图像
            valid_images = [img for img in image_batch if img is not None]
            if not valid_images:
                continue
                
            # 特征提取
            features = self._process_batch(valid_images)
            
            # 写入HDF5
            h5_indices = [i + idx for idx in valid_indices]
            self.dataset[h5_indices] = features


def imageid_to_file(img_id):
    path = "/media/yuchuxi/ST1/image_cache/"
    if img_id.find(".") > 4:
        path += img_id[0:4] + "/"
        return path + img_id[4:]
    else:
        return path + img_id


if __name__ == "__main__":
    if os.path.isfile("data/image_path_cache"):
        image_paths = joblib.load("data/image_path_cache")
    else:
        image_paths = [
            file
            for file in map(
                imageid_to_file, tqdm(joblib.load("data/image_stat").keys())
            )
            if os.path.isfile(file)
        ]
        joblib.dump(image_paths, "data/image_path_cache")

    output_directory = "data/features"

    extractor = CLIPFeatureExtractor(
        batch_size=512,       # 根据GPU显存调整
        target_size=224,      # CLIP标准输入尺寸
        h5_path="data/features.h5"
    )
    
    extractor.process_paths(image_paths)



    # # 张量注入
    # target_tensor = torch.randn(2, 10, 1024)  # 假设C=1024
    # injector = TensorInjector("features_cache")

    # modified = injector.inject_features(
    #     x=target_tensor,
    #     image_ids=["img1", "img2"],  # 对应文件名stem
    #     positions=[(0, 2), (1, 5)]  # 注入位置
    # )
