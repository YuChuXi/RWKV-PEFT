import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class ViTProj(nn.Module):
    def __init__(
        self, n_vit_layer: int, n_vit_embd: int, n_llm_embd: int, hidden_dim: int = 1024
    ):
        super().__init__()
        self.n_vit_layer = n_vit_layer

        # 参数堆叠初始化
        self.w1 = nn.Parameter(torch.Tensor(n_vit_layer, n_vit_embd, hidden_dim))
        self.w2 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim, n_llm_embd))
        self.b1 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim))
        self.b2 = nn.Parameter(torch.Tensor(n_vit_layer, n_llm_embd))

        # 参数初始化
        nn.init.kaiming_normal_(self.w1, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.w2, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

        # 添加激活函数和分组标准化
        self.act = nn.GELU()
        self.gn1 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入形状: (batch_size, n_vit_layer, n_vit_embd)
        输出形状: (batch_size, n_vit_layer, n_llm_embd)
        """
        batch_size = x.size(0)

        # 第一层投影 + 激活函数 + 分组标准化
        x = torch.einsum("ble,leh->blh", x, self.w1) + self.b1.unsqueeze(0)
        x = self.act(x)
        x = self.gn1(x)  # 输入形状: (batch_size, n_vit_layer, hidden_dim)

        # 第二层投影
        x = torch.einsum("blh,lho->blo", x, self.w2) + self.b2.unsqueeze(0)
        return x.view(batch_size, self.n_vit_layer, -1)


class ReViTProj(nn.Module):
    def __init__(
        self, n_vit_layer: int, n_llm_embd: int, n_vit_embd: int, hidden_dim: int = 1024
    ):
        super().__init__()
        self.n_vit_layer = n_vit_layer

        # 参数堆叠初始化
        self.w1 = nn.Parameter(torch.Tensor(n_vit_layer, n_llm_embd, hidden_dim))
        self.w2 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim, n_vit_embd))
        self.b1 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim))
        self.b2 = nn.Parameter(torch.Tensor(n_vit_layer, n_vit_embd))

        # 参数初始化
        nn.init.kaiming_normal_(self.w1, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.w2, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

        # 添加激活函数和分组标准化
        self.act = nn.GELU()
        self.gn1 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入形状: (batch_size, n_vit_layer, n_llm_embd)
        输出形状: (batch_size, n_vit_layer, n_vit_embd)
        """
        batch_size = x.size(0)

        # 第一层反投影 + 激活函数 + 分组标准化
        x = torch.einsum("ble,leh->blh", x, self.w1) + self.b1.unsqueeze(0)
        x = self.act(x)
        x = self.gn1(x)  # 输入形状: (batch_size, n_vit_layer, hidden_dim)

        # 第二层反投影
        x = torch.einsum("blh,lho->blo", x, self.w2) + self.b2.unsqueeze(0)
        return x.view(batch_size, self.n_vit_layer, -1)


class EmbeddingAndIMGProj(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        n_vit_embd: int,
        n_vit_layer: int,
        img_padding_idx: int,
        temperature: float = 0.07,
        recon_weight: float = 1.0,
        contrast_weight: float = 0.5,
        **kwargs,
    ):
        # 继承父类初始化
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=img_padding_idx,
            **kwargs,
        )

        self.n_vit_layer = n_vit_layer
        self.n_vit_embd = n_vit_embd
        self.embedding_dim = embedding_dim
        self.img_padding_idx = img_padding_idx

        self.vit_proj = ViTProj(
            n_vit_layer=n_vit_layer, n_vit_embd=n_vit_embd, n_llm_embd=embedding_dim
        )
        self.vit_reverse_proj = ReViTProj(
            n_vit_layer=n_vit_layer, n_llm_embd=embedding_dim, n_vit_embd=n_vit_embd
        )

        self.register_buffer("last_batch_indices", None)
        self.register_buffer(
            "last_token_indices", None
        )  # (batch_indices, token_indices)

        # 用于计算loss
        self.temperature = temperature
        self.recon_weight = recon_weight
        self.contrast_weight = contrast_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        vit_features_list: Optional[List[Dict[int, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        embeddings = super().forward(input_ids)

        if vit_features_list is None:
            return embeddings

        batch_indices, token_indices, vit_features = [], [], []

        for batch_idx, sample_dict in enumerate(vit_features_list):
            for start_idx, feat in sample_dict.items():
                feat = torch.tensor(
                    feat, dtype=embeddings.dtype, device=embeddings.device
                )

                # 动态计算有效插入长度
                max_len = input_ids.shape[1]
                valid_layers = min(self.n_vit_layer, max_len - start_idx)

                if valid_layers <= 0:
                    continue

                # 特征切片处理
                feat_slice = feat[:valid_layers]  # (valid_layers, n_vit_embd)
                # assert feat_slice.shape == (
                #     valid_layers,
                #     feat.shape[1],
                # ), f"Invalid feature shape {feat_slice.shape}"

                # 生成插入位置索引
                end_idx = start_idx + valid_layers
                current_batch = [batch_idx] * valid_layers
                current_tokens = list(range(start_idx, end_idx))

                # # 验证padding位置
                # padding_check = input_ids[batch_idx, start_idx:end_idx]
                # assert torch.all(
                #     padding_check == self.img_padding_idx
                # ), "Insert positions must be IMG_PADDING"

                batch_indices.extend(current_batch)
                token_indices.extend(current_tokens)
                vit_features.append(feat_slice)

        if vit_features:

            # 转换为张量索引
            batch_tensor = torch.tensor(batch_indices, device=input_ids.device)
            token_tensor = torch.tensor(token_indices, device=input_ids.device)

            vit_tensor = torch.cat(vit_features)  # (sum(layers), n_vit_embd)
            L = vit_tensor.size(0)
            padding_length = (math.ceil(L / self.n_vit_layer) * self.n_vit_layer) - L
            vit_tensor = F.pad(vit_tensor, (0, 0, 0, padding_length))  # (B * L, D)
            vit_tensor = vit_tensor.view(
                -1, self.n_vit_layer, self.n_vit_embd
            )  # (B, L, D)

            projected = self.vit_proj(vit_tensor)  # (b, layers, n_llm_embd)

            projected = projected.flatten(start_dim=0, end_dim=1)[:L]

            # 高效索引更新
            embeddings[batch_tensor, token_tensor] = projected
            self.last_batch_indices, self.last_token_indices = (
                batch_tensor,
                token_tensor,
            )

        return embeddings

    def decode_vit_features(self, outputs: torch.Tensor) -> torch.Tensor:
        assert (self.last_batch_indices is not None) and (
            self.last_token_indices is not None
        ), "Run forward with vit features first"

        # 考虑自回归模型的输出位移
        batch_indices, token_indices = self.last_batch_indices, self.last_token_indices
        shifted_tokens = torch.clamp(token_indices - 1, min=0)  # 处理序列起始位置

        # 提取输出特征并逆投影
        vit_outputs = outputs[batch_indices, shifted_tokens]  # (N, n_llm_embd)

        L = vit_outputs.size(0)
        padding_length = (math.ceil(L / self.n_vit_layer) * self.n_vit_layer) - L
        vit_outputs_padded = F.pad(vit_outputs, (0, 0, 0, padding_length))  # (B * L, D)
        vit_outputs_padded = vit_outputs_padded.view(
            -1, self.n_vit_layer, self.embedding_dim
        )  # (B, L, D)

        vit_recon = self.vit_reverse_proj(vit_outputs_padded)
        
        vit_recon = vit_recon.flatten(start_dim=0, end_dim=1)[:L]
        return vit_recon  # (B, L, D)

    def vit_reconstruction_loss(
        self,
        model_output: torch.Tensor,  # 模型输出张量 (batch_size, seq_len, hidden_dim)
        vit_features_list: List[Dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        多模态特征重建损失计算器
        参数：
            model_output: LLM的输出隐状态 (batch_size, seq_len, hidden_dim)
            vit_features_list: 原始ViT特征列表
            embed_module: 使用的嵌入投影模块实例
            temperature: 对比学习温度系数
            recon_weight: 特征重建损失权重
            contrast_weight: 对比学习损失权重
        返回：
            total_loss: 总损失值
            loss_dict: 各损失分量详情
        """
        # 类型检查
        assert model_output.dim() == 3, "Model output should be 3D tensor"

        # 解码ViT特征
        reconstructed = self.decode_vit_features(
            model_output
        )  # (total_layers, vit_dim)

        # 对齐原始特征
        batch_size, seq_len = model_output.shape[:2]
        original_features = []
        valid_mask = []

        # 遍历所有样本和插入位置
        for batch_idx, sample_dict in enumerate(vit_features_list):
            for start_idx, feat in sample_dict.items():

                feat = torch.tensor(
                    feat, dtype=reconstructed.dtype, device=reconstructed.device
                )
                # 有效性检查
                assert feat.shape[1] == self.n_vit_embd, "Feature dimension mismatch"

                # 计算实际可插入层数
                max_valid = min(feat.size(0), seq_len - start_idx)
                if max_valid <= 0:
                    continue

                # 保留有效特征
                valid_feat = feat[:max_valid]
                original_features.append(valid_feat)
                valid_mask.append(torch.ones(max_valid, dtype=torch.bool))

        # 无有效特征时返回零损失
        if not original_features:
            return torch.tensor(0.0, device=model_output.device), {}

        # 合并特征并创建掩码
        original = torch.cat(original_features, dim=0)
        valid_mask = torch.cat(valid_mask, dim=0)

        # 重建损失计算
        recon_loss = F.mse_loss(reconstructed[valid_mask], original)

        # 对比学习损失
        norm_recon = F.normalize(reconstructed[valid_mask], dim=-1)
        norm_original = F.normalize(original, dim=-1)
        logits = torch.einsum("nd,md->nm", norm_recon, norm_original) / self.temperature
        contrast_loss = F.cross_entropy(
            logits, torch.arange(logits.size(0), device=logits.device)
        )

        # 损失组合
        total_loss = (
            self.recon_weight * recon_loss + self.contrast_weight * contrast_loss
        )

        
        return total_loss, {
            "vit_total_loss": total_loss.detach(),
            "vit_recon_loss": recon_loss.detach(),
            "vit_contrast_loss": contrast_loss.detach(),
        }

    # 保持与原始Embedding兼容的方法
    @classmethod
    def from_pretrained(
        cls, embeddings: nn.Embedding, n_vit_embd: int, n_vit_layer: int
    ):
        """从现有Embedding实例创建兼容模块"""
        return cls(
            num_embeddings=embeddings.num_embeddings,
            embedding_dim=embeddings.embedding_dim,
            n_vit_embd=n_vit_embd,
            n_vit_layer=n_vit_layer,
            img_padding_idx=embeddings.padding_idx,
            _weight=embeddings.weight,
        )
