import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleMultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, head_dim=32, window_size=16,kernel_size_o=9):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        # self.kernel_size_o = kernel_size_o
        # 动态计算投影维度
        self.qkv_proj = nn.Conv2d(in_channels, num_heads * head_dim, 1)

        self.out_proj = nn.Conv2d(num_heads * head_dim, in_channels, kernel_size=kernel_size_o, stride=1, padding=0)

    def forward(self, x,attention,weight_map):
        B, C, H, W = x.size()

        # 生成可学习的QKV投影
        q = self.qkv_proj(weight_map)  # [B, 3*h*d, H, W]
        k = self.qkv_proj(attention)  # [B, 3*h*d, H, W]
        v = self.qkv_proj(x)  # [B, 3*h*d, H, W]
        # q, k, v = torch.chunk(qkv, 3, dim=1)  # 各 [B, h*d, H, W]

        H_pad, W_pad = self._window_partition2(q, self.window_size)

        # print("q, k, v shape after projection:", q.shape, k.shape, v.shape)

        # 窗口划分
        q = self._window_partition(q, self.window_size)  # [B*h, d, ws, ws]
        k = self._window_partition(k, self.window_size)
        v = self._window_partition(v, self.window_size)

        # print("q, k, v shape after window partition:", q.shape, k.shape, v.shape)

        # 重塑为多头形式
        q = q.view(B, self.num_heads, self.head_dim, -1, self.window_size ** 2).permute(0, 1, 3, 2, 4)  # [B, h, N, d, ws²]
        k = k.view(B, self.num_heads, self.head_dim, -1, self.window_size ** 2).permute(0, 1, 3, 2, 4)  # [B, h, N, d, ws²]
        v = v.view(B, self.num_heads, self.head_dim, -1, self.window_size ** 2).permute(0, 1, 3, 4, 2)  # [B, h, N, ws², d]

        # print("q, k, v shape after reshaping:", q.shape, k.shape, v.shape)

        # 窗口内注意力计算
        attn = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)  # [B, h, N, ws², ws²]
        attn = F.softmax(attn, dim=-1)

        # print("attn shape:", attn.shape)
        # print("v shape:", v.shape)
        # 聚合特征
        x = (attn @ v.transpose(-1, -2)).permute(0, 1, 4, 3, 2)  # [B, h, d, N, ws²]
        # print("x shape:", x.shape)
        x = x.reshape(B, self.num_heads * self.head_dim, H_pad, W_pad)

        # print("x shape before out_proj:", x.shape)
        output = self.out_proj(x)
        # print("output shape :", output.shape)
        # 最终投影
        return output

    def _window_partition(self, x, window_size):
        """
        将特征图划分为局部窗口
        输入: [B, C, H, W]
        输出: [B*num_windows, C, window_size, window_size]
        """
        B, C, H, W = x.size()

        # 如果 H 或 W 不能被 window_size 整除，进行填充
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # 右下填充

        H_pad, W_pad = H + pad_h, W + pad_w

        # 重塑为窗口形式
        x = x.view(B, C, H_pad // window_size, window_size, W_pad // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
        return windows

    def _window_partition2(self, x, window_size):
        """
        将特征图划分为局部窗口
        输入: [B, C, H, W]
        输出: [B*num_windows, C, window_size, window_size]
        """
        B, C, H, W = x.size()

        # 如果 H 或 W 不能被 window_size 整除，进行填充
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # 右下填充

        H_pad, W_pad = H + pad_h, W + pad_w

        # 重塑为窗口形式
        x = x.view(B, C, H_pad // window_size, window_size, W_pad // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
        return H_pad, W_pad

class BGMA(nn.Module):
    def __init__(self, in_channels, num_heads=8, head_dim=32, window_size=16,kernel_size_o=9):
        super(BGMA, self).__init__()
        self.window_size = window_size
        self.kernel_size_o = kernel_size_o
        self.in_channels = in_channels
        # self.kernel_size_o = kernel_size_o
        # 空间注意力（简化版）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

        # 多头注意力
        self.mha = FlexibleMultiHeadAttention(in_channels, num_heads, head_dim, window_size,kernel_size_o)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
            nn.ReLU()
        )

    def forward(self, x, weight_map):
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial = self.spatial_attn(torch.cat([max_pool, avg_pool], dim=1))

        # 通道注意力
        channel = self.channel_attn(x)

        # 注意力融合
        attention = spatial + channel

        # 带权重的多头注意力
        attn_out = self.mha(x,attention,weight_map)

        # 残差连接
        return self.fusion(torch.cat([x, attn_out], dim=1)) + x

# 计算可训练参数的函数
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 测试用例
if __name__ == "__main__":
    # 单通道测试
    x = torch.randn(2, 1, 62, 62)
    model = BGMA(in_channels=1, num_heads=4, head_dim=16,kernel_size_o=3)
    print("单通道输出形状:", model(x, x).shape)
    print("单通道输出形状:", model(x, x).shape)
    print("单通道模型可训练参数数量:", count_trainable_params(model))
    # 多通道测试
    x = torch.randn(16, 128, 62, 62)
    model = BGMA(in_channels=128, num_heads=8, head_dim=32,kernel_size_o=3)
    print("多通道输出形状:", model(x, x).shape)
    print("多通道输出形状:", model(x, x).shape)
    print("多通道模型可训练参数数量:", count_trainable_params(model))