import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional


class LayerwiseGroupNorm(nn.Module):
    def __init__(self, num_layers: int, num_cannels: int, eps: float = 1e-5):
        super().__init__()
        self.num_layers = num_layers
        self.num_cannels = num_cannels
        self.eps = eps
        # 每个层有独立的缩放和偏移参数
        self.weight = nn.Parameter(torch.ones(num_layers, 1))
        self.bias = nn.Parameter(torch.zeros(num_layers, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, layer, cannels = x.shape
        # 调整维度为 (layer, batch * cannels)
        x_reshaped = x.permute(1, 0, 2).contiguous().view(layer, -1)
        # 计算均值和方差
        mean = x_reshaped.mean(dim=1, keepdim=True)
        var = x_reshaped.var(dim=1, keepdim=True, unbiased=False)
        # 标准化
        x_normalized = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        # 应用缩放和偏移
        x_normalized = x_normalized * self.weight + self.bias
        # 恢复原始形状
        x_out = x_normalized.view(layer, batch, cannels).permute(1, 0, 2)
        return x_out
    


class BaseProjector(nn.Module):
    """投影基类（正向/反向共享结构）"""

    def __init__(self, n_layers, in_dim, out_dim, hidden_dim=1024, reverse=False):
        super().__init__()
        self.n_layers = n_layers
        self.reverse = reverse

        # 投影参数定义
        self.w1 = nn.Parameter(torch.Tensor(n_layers, in_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.Tensor(n_layers, hidden_dim, out_dim))

        # 偏置项
        self.b1 = nn.Parameter(torch.zeros(n_layers, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_layers, out_dim))
        self.b_shortcut = nn.Parameter(torch.zeros(n_layers, out_dim))

        # 初始化参数
        nn.init.kaiming_normal_(self.w1, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.w2, mode="fan_in", nonlinearity="linear")

        # 共享网络组件
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.0)
        self.norm = LayerwiseGroupNorm(n_layers, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主分支处理流程
        x = torch.einsum("ble,leh->blh", x, self.w1) + self.b1.unsqueeze(0)
        x = self.act(x)  # 后接激活函数
        x = self.drop(x)
        x = torch.einsum("blh,lho->blo", x, self.w2) + self.b2.unsqueeze(0)
        return x + self.norm(x)  # 特征增强


class ViTProj(BaseProjector):
    """视觉特征投影（ViT->LLM）"""

    def __init__(
        self, n_vit_layer: int, n_vit_embd: int, n_llm_embd: int, hidden_dim: int = 1024
    ):
        super().__init__(
            n_layers=n_vit_layer,
            in_dim=n_vit_embd,
            out_dim=n_llm_embd,
            hidden_dim=hidden_dim,
            reverse=False,
        )


class ReViTProj(BaseProjector):
    """逆向特征投影（LLM->ViT）"""

    def __init__(
        self, n_vit_layer: int, n_llm_embd: int, n_vit_embd: int, hidden_dim: int = 1024
    ):
        super().__init__(
            n_layers=n_vit_layer,
            in_dim=n_llm_embd,
            out_dim=n_vit_embd,
            hidden_dim=hidden_dim,
            reverse=True,
        )


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
        device=None,
        dtype=None,
        **kwargs,
    ):
        # 继承父类初始化
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=img_padding_idx,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        self.n_vit_layer = n_vit_layer
        self.n_vit_embd = n_vit_embd
        self.embedding_dim = embedding_dim
        self.img_padding_idx = img_padding_idx

        self.vit_proj = ViTProj(
            n_vit_layer=n_vit_layer,
            n_vit_embd=n_vit_embd,
            n_llm_embd=embedding_dim,
            hidden_dim=embedding_dim,
        )
        self.vit_reverse_proj = ReViTProj(
            n_vit_layer=n_vit_layer,
            n_llm_embd=embedding_dim,
            n_vit_embd=n_vit_embd,
            hidden_dim=embedding_dim,
        )

        self.register_buffer("model_input", None)

        # 用于计算loss
        self.temperature = temperature
        self.recon_weight = recon_weight
        self.contrast_weight = contrast_weight

    def arrange_vit_feature(self, vit_features_list, max_len):
        batch_indices, token_indices, vit_features = [], [], []
        for batch_idx, sample_dict in enumerate(vit_features_list):
            for start_idx, feat in sample_dict.items():
                feat = torch.tensor(
                    feat, dtype=self.weight.dtype, device=self.weight.device
                )

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
            batch_tensor = torch.tensor(batch_indices, device=self.weight.device)
            token_tensor = torch.tensor(token_indices, device=self.weight.device)
            vit_tensor = torch.cat(vit_features)  # (sum(layers), n_vit_embd)
            return batch_tensor, token_tensor, vit_tensor
        return None, None, None

    def forward(
        self,
        input_ids: torch.Tensor,
        vit_features_list: Optional[List[Dict[int, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        embeddings = super().forward(input_ids)

        if vit_features_list is None:
            return embeddings

        # 动态计算有效插入长度
        max_len = input_ids.shape[1]
        batch_tensor, token_tensor, vit_tensor = self.arrange_vit_feature(
            vit_features_list, max_len
        )

        if vit_tensor is not None:
            model_input = self.encode_vit_features(vit_tensor)
            embeddings[batch_tensor, token_tensor] = model_input
            self.model_input = model_input
            self.last_batch_indices, self.last_token_indices = (
                batch_tensor,
                token_tensor,
            )

        return embeddings

    def encode_vit_features(self, vit_inputs: torch.Tensor) -> torch.Tensor:
        L = vit_inputs.size(0)
        padding_length = (math.ceil(L / self.n_vit_layer) * self.n_vit_layer) - L
        vit_inputs_padded = F.pad(vit_inputs, (0, 0, 0, padding_length))  # (B * L, D)
        vit_inputs_padded = vit_inputs_padded.view(
            -1, self.n_vit_layer, self.n_vit_embd
        )  # (B, L, D)

        projected = self.vit_proj(vit_inputs_padded)  # (b, L, n_llm_embd)

        return projected.flatten(start_dim=0, end_dim=1)[:L]  # (B * L, D)

    def decode_vit_features(self, vit_outputs: torch.Tensor) -> torch.Tensor:
        L = vit_outputs.size(0)
        padding_length = (math.ceil(L / self.n_vit_layer) * self.n_vit_layer) - L
        vit_outputs_padded = F.pad(vit_outputs, (0, 0, 0, padding_length))  # (B * L, D)
        vit_outputs_padded = vit_outputs_padded.view(
            -1, self.n_vit_layer, self.embedding_dim
        )  # (B, L, D)

        reprojected = self.vit_reverse_proj(vit_outputs_padded)

        return reprojected.flatten(start_dim=0, end_dim=1)[:L]  # (B * L, D)

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
        返回：
            total_loss: 总损失值
            loss_dict: 各损失分量详情
        """
        # 类型检查
        assert model_output.dim() == 3, "Model output should be 3D tensor"

        # 解码ViT特征

        # 考虑自回归模型的输出位移
        max_len = model_output.shape[1]
        batch_tensor, token_tensor, vit_input = self.arrange_vit_feature(
            vit_features_list, max_len
        )
        # 无有效特征时返回零损失
        if vit_input is None:
            return torch.tensor(0.0, device=model_output.device), {}

        model_input = self.model_input

        # 提取输出特征并逆投影
        shifted_tokens = torch.clamp(token_tensor - 1, min=0)  # 处理序列起始位置
        model_output = model_output[batch_tensor, shifted_tokens]  # (N, n_llm_embd)

        # 最终vit特征
        vit_output = self.decode_vit_features(model_output)  # (total_layers, vit_dim)
        vit_output_skip_rwkv = self.decode_vit_features(
            model_input
        )  # (total_layers, vit_dim)

        # 重建损失计算
        vit_recon_loss = F.l1_loss(vit_output, vit_input)  # 值太小
        vit_recon_loss_skip_rwkv = F.l1_loss(vit_output_skip_rwkv, vit_input)  # 值太小
        vit_emb_loss = F.mse_loss(model_output, model_input)

        total_loss = (
            vit_recon_loss * 5 + vit_recon_loss_skip_rwkv * 10 + vit_emb_loss * 0.2
        )

        # # 对比学习损失
        # norm_recon = F.normalize(reconstructed[valid_mask], dim=-1)
        # norm_original = F.normalize(original, dim=-1)
        # logits = torch.einsum("nd,md->nm", norm_recon, norm_original) / self.temperature
        # contrast_loss = F.cross_entropy(
        #     logits, torch.arange(logits.size(0), device=logits.device)
        # )

        # # 损失组合
        # total_loss = (
        #     self.recon_weight * recon_loss + self.contrast_weight * contrast_loss
        # )

        return total_loss, {
            "vit_total_loss": total_loss.detach(),
            "vit_recon_loss": vit_recon_loss.detach(),
            "vit_recon_loss_skip_rwkv": vit_recon_loss_skip_rwkv.detach(),
            "vit_emb_loss": vit_emb_loss.detach(),
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
