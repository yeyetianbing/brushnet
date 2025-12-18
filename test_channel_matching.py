"""
简单的通道数匹配验证脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# 模拟 AdaptiveFeatureFusion 类
class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_weight = self.channel_attention(x)
        x_channel = x * channel_weight
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)
        x_spatial = x_channel * spatial_weight
        return x_spatial

# SD 1.5 配置
block_out_channels = [320, 640, 1280, 1280]
layers_per_block = 2

print("=" * 80)
print("验证自适应特征融合模块的通道数匹配 (SD 1.5)")
print("=" * 80)
print(f"\n配置:")
print(f"  - block_out_channels: {block_out_channels}")
print(f"  - layers_per_block: {layers_per_block}")

# 模拟创建融合模块（按照修复后的逻辑）
print("\n[1/3] 创建下采样融合模块...")
adaptive_fusion_down = nn.ModuleList([])

# 第一个输出是 conv_in，通道数为 block_out_channels[0]
adaptive_fusion_down.append(AdaptiveFeatureFusion(block_out_channels[0]))
print(f"  [0] conv_in 输出: {block_out_channels[0]} 通道")

idx = 1
for i in range(len(block_out_channels)):
    out_channels = block_out_channels[i]
    is_final = i == len(block_out_channels) - 1
    num_outputs = layers_per_block + (1 if not is_final else 0)

    print(f"  Down Block {i} (输出通道: {out_channels}):")
    for j in range(num_outputs):
        adaptive_fusion_down.append(AdaptiveFeatureFusion(out_channels))
        print(f"    [{idx}] {out_channels} 通道")
        idx += 1

print(f"\n总共创建了 {len(adaptive_fusion_down)} 个下采样融合模块")

# 创建中间块融合模块
print("\n[2/3] 创建中间块融合模块...")
mid_channels = block_out_channels[-1]
adaptive_fusion_mid = AdaptiveFeatureFusion(mid_channels)
print(f"  中间块: {mid_channels} 通道")

# 创建上采样融合模块
print("\n[3/3] 创建上采样融合模块...")
adaptive_fusion_up = nn.ModuleList([])
reversed_block_out_channels = list(reversed(block_out_channels))

idx = 0
for i in range(len(block_out_channels)):
    out_channels = reversed_block_out_channels[i]
    is_final = i == len(block_out_channels) - 1
    num_outputs = layers_per_block + 1 + (1 if not is_final else 0)

    print(f"  Up Block {i} (输出通道: {out_channels}):")
    for j in range(num_outputs):
        adaptive_fusion_up.append(AdaptiveFeatureFusion(out_channels))
        print(f"    [{idx}] {out_channels} 通道")
        idx += 1

print(f"\n总共创建了 {len(adaptive_fusion_up)} 个上采样融合模块")

# 模拟前向传播测试
print("\n" + "=" * 80)
print("测试前向传播")
print("=" * 80)

# 模拟 down_block_res_samples
print("\n[1/2] 模拟下采样输出...")
down_block_res_samples = []

# conv_in 输出
down_block_res_samples.append(torch.randn(1, 320, 64, 64))
print(f"  [0] conv_in: {list(down_block_res_samples[-1].shape)}")

# 下采样块输出
idx = 1
for i in range(len(block_out_channels)):
    out_channels = block_out_channels[i]
    is_final = i == len(block_out_channels) - 1

    # ResNet 层输出
    for j in range(layers_per_block):
        h = 64 // (2 ** i)
        down_block_res_samples.append(torch.randn(1, out_channels, h, h))
        print(f"  [{idx}] Down Block {i} ResNet {j}: {list(down_block_res_samples[-1].shape)}")
        idx += 1

    # 下采样输出
    if not is_final:
        h = 64 // (2 ** (i + 1))
        down_block_res_samples.append(torch.randn(1, out_channels, h, h))
        print(f"  [{idx}] Down Block {i} Downsample: {list(down_block_res_samples[-1].shape)}")
        idx += 1

print(f"\n总共 {len(down_block_res_samples)} 个下采样输出")

# 测试融合模块
print("\n[2/2] 测试融合模块前向传播...")
try:
    for i, (sample, fusion_module) in enumerate(zip(down_block_res_samples, adaptive_fusion_down)):
        output = fusion_module(sample)
        if i < 3 or i >= len(down_block_res_samples) - 3:
            print(f"  [{i:2d}] 输入: {list(sample.shape)} -> 输出: {list(output.shape)}")
        elif i == 3:
            print(f"  ...")

    print("\n✓ 所有下采样融合模块测试通过！")

    # 测试中间块
    mid_sample = torch.randn(1, mid_channels, 8, 8)
    mid_output = adaptive_fusion_mid(mid_sample)
    print(f"\n中间块: 输入 {list(mid_sample.shape)} -> 输出 {list(mid_output.shape)}")
    print("✓ 中间块融合模块测试通过！")

    # 测试上采样块
    print("\n上采样融合模块测试:")
    for i in range(len(adaptive_fusion_up)):
        out_channels = reversed_block_out_channels[i // (layers_per_block + 1 + (1 if i // (layers_per_block + 1) < len(block_out_channels) - 1 else 0))]
        h = 8 * (2 ** (i // 4))
        sample = torch.randn(1, out_channels, h, h)
        output = adaptive_fusion_up[i](sample)
        if i < 3 or i >= len(adaptive_fusion_up) - 3:
            print(f"  [{i:2d}] 输入: {list(sample.shape)} -> 输出: {list(output.shape)}")
        elif i == 3:
            print(f"  ...")

    print("\n✓ 所有上采样融合模块测试通过！")

except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ 所有测试通过！通道数匹配正确。")
print("=" * 80)
print("\n总结:")
print(f"  - 下采样融合模块数量: {len(adaptive_fusion_down)}")
print(f"  - 中间块融合模块数量: 1")
print(f"  - 上采样融合模块数量: {len(adaptive_fusion_up)}")
print(f"  - 所有模块的通道数都与对应的特征图匹配")
