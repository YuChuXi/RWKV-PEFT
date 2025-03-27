import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
class ViTProj(nn.Module):
    def __init__(
        self,
        n_vit_layer: int,
        n_vit_embd: int,
        n_llm_embd: int,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.n_vit_layer = n_vit_layer

        # 主路径参数
        self.w1 = nn.Parameter(torch.Tensor(n_vit_layer, n_vit_embd, hidden_dim))
        self.w2 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim, n_llm_embd))
        self.b1 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim))
        self.b2 = nn.Parameter(torch.Tensor(n_vit_layer, n_llm_embd))

        # 残差路径参数（新增）
        self.w_res = nn.Parameter(torch.Tensor(n_vit_layer, n_vit_embd, n_llm_embd))
        self.b_res = nn.Parameter(torch.Tensor(n_vit_layer, n_llm_embd))

        # 参数初始化
        for w in [self.w1, self.w2, self.w_res]:
            nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="linear")
        for b in [self.b1, self.b2, self.b_res]:
            nn.init.zeros_(b)

        # 标准化层调整分组数（保持每个层独立归一化）
        self.gn1 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)
        self.gn2 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 输入形状: (b, l, e_vit), 输出形状: (b, l, e_llm) """
        # 残差路径投影
        res = torch.einsum("ble,leh->blh", x, self.w_res) + self.b_res.unsqueeze(0)
        
        # 主路径处理
        # 第一层投影
        x = torch.einsum("ble,leh->blh", x, self.w1) + self.b1.unsqueeze(0)
        x = self.gn1(x)  # 形状保持(b, l, h)
        x = self.act(x)
        x = self.drop(x)
        
        # 第二层投影
        x = torch.einsum("blh,lho->blo", x, self.w2) + self.b2.unsqueeze(0)
        x = self.gn2(x)  # 形状变为(b, l, e_llm)
        
        # 合并残差
        return res + x


class ReViTProj(nn.Module):
    def __init__(
        self,
        n_vit_layer: int,
        n_llm_embd: int,
        n_vit_embd: int,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.n_vit_layer = n_vit_layer

        # 主路径参数
        self.w1 = nn.Parameter(torch.Tensor(n_vit_layer, n_llm_embd, hidden_dim))
        self.w2 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim, n_vit_embd))
        self.b1 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim))
        self.b2 = nn.Parameter(torch.Tensor(n_vit_layer, n_vit_embd))

        # 残差路径参数（新增）
        self.w_res = nn.Parameter(torch.Tensor(n_vit_layer, n_llm_embd, n_vit_embd))
        self.b_res = nn.Parameter(torch.Tensor(n_vit_layer, n_vit_embd))

        # 参数初始化
        for w in [self.w1, self.w2, self.w_res]:
            nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="linear")
        for b in [self.b1, self.b2, self.b_res]:
            nn.init.zeros_(b)

        # 标准化层调整分组数
        self.gn1 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)
        self.gn2 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 输入形状: (b, l, e_llm), 输出形状: (b, l, e_vit) """
        # 残差路径投影
        res = torch.einsum("ble,leh->blh", x, self.w_res) + self.b_res.unsqueeze(0)
        
        # 主路径处理
        # 第一层反投影
        x = torch.einsum("ble,leh->blh", x, self.w1) + self.b1.unsqueeze(0)
        x = self.gn1(x)  # 形状保持(b, l, h)
        x = self.act(x)
        x = self.drop(x)
        
        # 第二层反投影
        x = torch.einsum("blh,lho->blo", x, self.w2) + self.b2.unsqueeze(0)
        x = self.gn2(x)  # 形状变为(b, l, e_vit)
        
        # 合并残差
        return res + x

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

        proj_hidden = 1024
        self.vit_proj = ViTProj(
            n_vit_layer=n_vit_layer,
            n_vit_embd=n_vit_embd,
            n_llm_embd=embedding_dim,
            hidden_dim=proj_hidden,
        )
        self.vit_reverse_proj = ReViTProj(
            n_vit_layer=n_vit_layer,
            n_llm_embd=embedding_dim,
            n_vit_embd=n_vit_embd,
            hidden_dim=proj_hidden,
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
