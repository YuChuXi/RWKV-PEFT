import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

big_vit_proj = True # FIXME 使用配置

if big_vit_proj:
    class ViTProj(nn.Module):
        def __init__(
            self, 
            n_vit_layer: int,
            n_vit_embd: int,
            n_llm_embd: int,
            hidden_dim: int = 1024,
            expansion: int = 4,
            num_groups: int = 8
        ):
            super().__init__()
            assert hidden_dim == n_llm_embd, "hidden_dim must equal n_llm_embd"
            self.n_vit_layer = n_vit_layer
            self.expansion = expansion

            # 参数结构：动态门控+扩展投影
            self.w_gate = nn.Parameter(torch.Tensor(n_vit_layer, n_vit_embd, hidden_dim*2))
            self.w_proj = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim*2, hidden_dim*expansion))
            self.w_attn = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim*expansion, hidden_dim))
            self.b_gate = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim*2))
            self.b_proj = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim*expansion))
            self.b_attn = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim))

            # 跨层注意力机制
            self.layer_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

            # 归一化层优化
            self.gn_input = nn.GroupNorm(min(num_groups, n_vit_layer), n_vit_layer)
            self.gn_hidden = nn.GroupNorm(min(num_groups, n_vit_layer), n_vit_layer)
            self.ln_output = nn.LayerNorm(hidden_dim)

            # 激活与正则化
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.1)
            
            # 初始化
            for w in [self.w_gate, self.w_proj, self.w_attn]:
                nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='linear')
            for b in [self.b_gate, self.b_proj, self.b_attn]:
                nn.init.zeros_(b)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            输入形状: (B, L, E_vit)
            输出形状: (B, L, E_llm)
            """
            B, L, _ = x.shape
            
            # 输入归一化
            x = self.gn_input(x)
            
            # 动态门控投影
            gate = torch.einsum('ble,leh->blh', x, self.w_gate) + self.b_gate.unsqueeze(0)
            gate, x_proj = torch.chunk(gate, 2, dim=-1)
            gate = torch.sigmoid(gate)
            x = x_proj * gate
            
            # 扩展投影
            x = self.act(torch.einsum('blh,leh->blh', x, self.w_proj) + self.b_proj.unsqueeze(0))
            x = self.gn_hidden(x)
            x = self.drop(x)
            
            # 跨层注意力
            x_attn, _ = self.layer_attn(
                x.view(B*L, 1, -1), 
                x.view(B*L, 1, -1),
                x.view(B*L, 1, -1)
            )
            x = x + x_attn.view(B, L, -1)
            
            # 压缩投影
            x = self.ln_output(
                torch.einsum('blh,leh->blh', x, self.w_attn) + self.b_attn.unsqueeze(0)
            )
            return x

    class ReViTProj(nn.Module):
        def __init__(
            self,
            n_vit_layer: int,
            n_llm_embd: int,
            n_vit_embd: int,
            hidden_dim: int = 1024,
            expansion: int = 2,
            num_groups: int = 8
        ):
            super().__init__()
            assert hidden_dim == n_llm_embd, "hidden_dim must equal n_llm_embd"
            self.n_vit_layer = n_vit_layer
            
            # 逆向投影参数
            self.w1 = nn.Parameter(torch.Tensor(n_vit_layer, n_llm_embd, hidden_dim*expansion))
            self.w2 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim*expansion, hidden_dim))
            self.w3 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim, n_vit_embd))
            self.b1 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim*expansion))
            self.b2 = nn.Parameter(torch.Tensor(n_vit_layer, hidden_dim))
            self.b3 = nn.Parameter(torch.Tensor(n_vit_layer, n_vit_embd))

            # 深度可分离卷积增强局部性
            self.depth_conv = nn.Conv1d(
                in_channels=n_vit_layer,
                out_channels=n_vit_layer,
                kernel_size=3,
                padding=1,
                groups=n_vit_layer
            )
            
            # 归一化层
            self.gn1 = nn.GroupNorm(min(num_groups, n_vit_layer), n_vit_layer)
            self.gn2 = nn.GroupNorm(min(num_groups, n_vit_layer), n_vit_layer)
            self.ln = nn.LayerNorm(n_vit_embd)
            
            # 激活与正则化
            self.act = nn.SiLU()  # 实验性激活函数
            self.drop = nn.Dropout(0.1)
            
            # 初始化
            for w in [self.w1, self.w2, self.w3]:
                nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='linear')
            for b in [self.b1, self.b2, self.b3]:
                nn.init.zeros_(b)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            输入形状: (B, L, E_llm)
            输出形状: (B, L, E_vit)
            """
            # 第一阶段投影
            x = self.act(
                torch.einsum('ble,leh->blh', x, self.w1) + self.b1.unsqueeze(0)
            )
            x = self.gn1(x)
            
            # 深度卷积增强
            x = x + self.depth_conv(x.transpose(1,2)).transpose(1,2)
            
            # 第二阶段投影
            x = self.act(
                torch.einsum('blh,leh->blh', x, self.w2) + self.b2.unsqueeze(0)
            )
            x = self.gn2(x)
            x = self.drop(x)
            
            # 最终投影
            x = self.ln(
                torch.einsum('blh,leh->ble', x, self.w3) + self.b3.unsqueeze(0)
            )
            return x

else:
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
            self.drop = nn.Dropout(0.1)
            self.gn0 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)
            self.gn1 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            输入形状: (batch_size, n_vit_layer, n_vit_embd)
            输出形状: (batch_size, n_vit_layer, n_llm_embd)
            """
            batch_size = x.size(0)

            # 第一层投影
            x = self.gn0(x)
            x = self.act(x)
            x = torch.einsum("ble,leh->blh", x, self.w1) + self.b1.unsqueeze(0)
            x = self.gn1(x)  # 输入形状: (batch_size, n_vit_layer, hidden_dim)
            xn = self.act(x)
            xn = self.drop(x)

            # 第二层残差投影
            xn = torch.einsum("blh,lho->blo", x, self.w2) + self.b2.unsqueeze(0)
            x = x + xn
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
            self.drop = nn.Dropout(0.1)
            self.gn0 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)
            self.gn1 = nn.GroupNorm(num_groups=n_vit_layer, num_channels=n_vit_layer)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            输入形状: (batch_size, n_vit_layer, n_llm_embd)
            输出形状: (batch_size, n_vit_layer, n_vit_embd)
            """
            batch_size = x.size(0)

            # 第一层残差反投影
            x = self.gn0(x)
            x = self.act(x)
            nx = torch.einsum("ble,leh->blh", x, self.w1) + self.b1.unsqueeze(0)
            nx = self.gn1(nx)  # 输入形状: (batch_size, n_vit_layer, hidden_dim)
            nx = self.act(nx)
            nx = self.drop(nx)
            x = x + nx
            
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

        proj_hidden = 2560
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
            vit_recon_loss * 5 + vit_recon_loss_skip_rwkv * 5 + vit_emb_loss * 0.2
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
