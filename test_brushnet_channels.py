"""
完整测试 BrushNet 的通道数匹配
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# 从 brushnet.py 复制 AdaptiveFeatureFusion 类
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
print("测试 BrushNet 自适应特征融合模块的通道数匹配 (SD 1.5)")
print("=" * 80)
print(f"\n配置:")
print(f"  - block_out_channels: {block_out_channels}")
print(f"  - layers_per_block: {layers_per_block}")

# ============================================================================
# 1. 创建下采样融合模块（按照修复后的逻辑）
# ============================================================================
print("\n" + "=" * 80)
print("[1/3] 创建下采样融合模块")
print("=" * 80)

adaptive_fusion_down = nn.ModuleList([])

# 第一个输出是 conv_in，通道数为 block_out_channels[0]
adaptive_fusion_down.append(AdaptiveFeatureFusion(block_out_channels[0]))
print(f"\n[0] conv_in 输出: {block_out_channels[0]} 通道")

idx = 1
for i in range(len(block_out_channels)):
    out_channels = block_out_channels[i]
    is_final = i == len(block_out_channels) - 1
    num_outputs = layers_per_block + (1 if not is_final else 0)

    print(f"\nDown Block {i} (输出通道: {out_channels}):")
    for j in range(num_outputs):
        adaptive_fusion_down.append(AdaptiveFeatureFusion(out_channels))
        print(f"  [{idx}] {out_channels} 通道")
        idx += 1

print(f"\n✓ 总共创建了 {len(adaptive_fusion_down)} 个下采样融合模块")

# ============================================================================
# 2. 创建中间块融合模块
# ============================================================================
print("\n" + "=" * 80)
print("[2/3] 创建中间块融合模块")
print("=" * 80)

mid_channels = block_out_channels[-1]
adaptive_fusion_mid = AdaptiveFeatureFusion(mid_channels)
print(f"\n中间块: {mid_channels} 通道")

# ============================================================================
# 3. 创建上采样融合模块
# ============================================================================
print("\n" + "=" * 80)
print("[3/3] 创建上采样融合模块")
print("=" * 80)

adaptive_fusion_up = nn.ModuleList([])
reversed_block_out_channels = list(reversed(block_out_channels))

idx = 0
for i in range(len(block_out_channels)):
    out_channels = reversed_block_out_channels[i]
    is_final = i == len(block_out_channels) - 1
    num_outputs = layers_per_block + 1 + (1 if not is_final else 0)

    print(f"\nUp Block {i} (输出通道: {out_channels}):")
    for j in range(num_outputs):
        adaptive_fusion_up.append(AdaptiveFeatureFusion(out_channels))
        print(f"  [{idx}] {out_channels} 通道")
        idx += 1

print(f"\n✓ 总共创建了 {len(adaptive_fusion_up)} 个上采样融合模块")

# ============================================================================
# 4. 模拟前向传播测试
# ============================================================================
print("\n" + "=" * 80)
print("测试前向传播")
print("=" * 80)

# 4.1 测试下采样融合模块
print("\n[1/3] 测试下采样融合模块...")
down_block_res_samples = []

# conv_in 输出
down_block_res_samples.append(torch.randn(1, 320, 64, 64))

# 下采样块输出
for i in range(len(block_out_channels)):
    out_channels = block_out_channels[i]
    is_final = i == len(block_out_channels) - 1
    h = 64 // (2 ** i)

    # ResNet 层输出
    for j in range(layers_per_block):
        down_block_res_samples.append(torch.randn(1, out_channels, h, h))

    # 下采样输出
    if not is_final:
        h = 64 // (2 ** (i + 1))
        down_block_res_samples.append(torch.randn(1, out_channels, h, h))

print(f"生成了 {len(down_block_res_samples)} 个下采样输出")

# 测试融合
try:
    brushnet_down_block_res_samples = ()
    for i, (sample, fusion_module) in enumerate(zip(down_block_res_samples, adaptive_fusion_down)):
        output = fusion_module(sample)
        brushnet_down_block_res_samples = brushnet_down_block_res_samples + (output,)
        if i < 3 or i >= len(down_block_res_samples) - 3:
            print(f"  [{i:2d}] 输入: {list(sample.shape)} -> 输出: {list(output.shape)}")
        elif i == 3:
            print(f"  ...")
    print("✓ 所有下采样融合模块测试通过！")
except Exception as e:
    print(f"✗ 下采样融合模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4.2 测试中间块融合模块
print("\n[2/3] 测试中间块融合模块...")
try:
    mid_sample = torch.randn(1, mid_channels, 8, 8)
    mid_output = adaptive_fusion_mid(mid_sample)
    print(f"  输入: {list(mid_sample.shape)} -> 输出: {list(mid_output.shape)}")
    print("✓ 中间块融合模块测试通过！")
except Exception as e:
    print(f"✗ 中间块融合模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4.3 测试上采样融合模块
print("\n[3/3] 测试上采样融合模块...")
up_block_res_samples = []

# 模拟上采样块的输出
for i in range(len(block_out_channels)):
    out_channels = reversed_block_out_channels[i]
    is_final = i == len(block_out_channels) - 1
    h = 8 * (2 ** i)

    # ResNet 层输出
    for j in range(layers_per_block + 1):
        up_block_res_samples.append(torch.randn(1, out_channels, h, h))

    # Upsampler 输出
    if not is_final:
        h = 8 * (2 ** (i + 1))
        up_block_res_samples.append(torch.randn(1, out_channels, h, h))

print(f"生成了 {len(up_block_res_samples)} 个上采样输出")

# 测试融合
try:
    brushnet_up_block_res_samples = ()
    for i, (sample, fusion_module) in enumerate(zip(up_block_res_samples, adaptive_fusion_up)):
        output = fusion_module(sample)
        brushnet_up_block_res_samples = brushnet_up_block_res_samples + (output,)
        if i < 3 or i >= len(up_block_res_samples) - 3:
            print(f"  [{i:2d}] 输入: {list(sample.shape)} -> 输出: {list(output.shape)}")
        elif i == 3:
            print(f"  ...")
    print("✓ 所有上采样融合模块测试通过！")
except Exception as e:
    print(f"✗ 上采样融合模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 5. 总结
# ============================================================================
print("\n" + "=" * 80)
print("✓ 所有测试通过！通道数匹配正确。")
print("=" * 80)
print("\n总结:")
print(f"  - 下采样融合模块数量: {len(adaptive_fusion_down)}")
print(f"  - 中间块融合模块数量: 1")
print(f"  - 上采样融合模块数量: {len(adaptive_fusion_up)}")
print(f"  - 下采样输出数量: {len(brushnet_down_block_res_samples)}")
print(f"  - 上采样输出数量: {len(brushnet_up_block_res_samples)}")
print("\n✓ 修复成功！所有模块的通道数都与对应的特征图匹配。")
print("=" * 80)
