"""
调试通道数不匹配问题
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# SD 1.5 配置
block_out_channels = [320, 640, 1280, 1280]
layers_per_block = 2

print("=" * 80)
print("调试通道数不匹配问题")
print("=" * 80)

# 模拟 brushnet_up_blocks 的创建（来自 brushnet.py 第441-449行）
print("\n[1/2] 创建 brushnet_up_blocks...")
brushnet_up_blocks = nn.ModuleList([])
reversed_block_out_channels = list(reversed(block_out_channels))

for i in range(len(block_out_channels)):
    is_final_block = i == len(block_out_channels) - 1
    output_channel = reversed_block_out_channels[i]

    print(f"\nUp Block {i} (output_channel={output_channel}):")

    # layers_per_block+1 个 ResNet 输出
    for j in range(layers_per_block + 1):
        brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        brushnet_up_blocks.append(brushnet_block)
        print(f"  [{len(brushnet_up_blocks)-1}] ResNet {j}: {output_channel} 通道")

    # 可能的 upsampler 输出
    if not is_final_block:
        brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        brushnet_up_blocks.append(brushnet_block)
        print(f"  [{len(brushnet_up_blocks)-1}] Upsampler: {output_channel} 通道")

print(f"\n总共 {len(brushnet_up_blocks)} 个 brushnet_up_blocks")

# 打印每个 brushnet_up_block 的通道数
print("\n[2/2] brushnet_up_blocks 的通道数:")
for i, block in enumerate(brushnet_up_blocks):
    in_ch = block.in_channels
    out_ch = block.out_channels
    print(f"  [{i:2d}] in={in_ch:4d}, out={out_ch:4d}")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
