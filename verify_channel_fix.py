"""
验证自适应特征融合模块的通道数是否正确匹配
"""
import torch
from src.diffusers.models.brushnet import BrushNetModel

# SD 1.5 配置
config = {
    "in_channels": 4,
    "conditioning_channels": 5,
    "block_out_channels": [320, 640, 1280, 1280],  # SD 1.5
    "layers_per_block": 2,
    "down_block_types": ["DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"],
    "mid_block_type": "UNetMidBlock2D",
    "up_block_types": ["UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"],
}

print("=" * 80)
print("验证自适应特征融合模块的通道数匹配")
print("=" * 80)
print(f"\n配置: SD 1.5")
print(f"  - block_out_channels: {config['block_out_channels']}")
print(f"  - layers_per_block: {config['layers_per_block']}")

# 创建 BrushNet 模型
print("\n[1/4] 创建 BrushNet 模型...")
try:
    brushnet = BrushNetModel(**config)
    print("✓ 模型创建成功")
except Exception as e:
    print(f"✗ 模型创建失败: {e}")
    exit(1)

# 检查融合模块数量
print("\n[2/4] 检查融合模块数量...")
num_down_fusion = len(brushnet.adaptive_fusion_down)
num_mid_fusion = 1
num_up_fusion = len(brushnet.adaptive_fusion_up)

print(f"  - 下采样融合模块: {num_down_fusion}")
print(f"  - 中间块融合模块: {num_mid_fusion}")
print(f"  - 上采样融合模块: {num_up_fusion}")

# 检查 brushnet_down_blocks 数量
num_brushnet_down = len(brushnet.brushnet_down_blocks)
num_brushnet_up = len(brushnet.brushnet_up_blocks)
print(f"  - brushnet_down_blocks: {num_brushnet_down}")
print(f"  - brushnet_up_blocks: {num_brushnet_up}")

if num_down_fusion != num_brushnet_down:
    print(f"✗ 错误: 下采样融合模块数量 ({num_down_fusion}) 与 brushnet_down_blocks ({num_brushnet_down}) 不匹配")
    exit(1)

if num_up_fusion != num_brushnet_up:
    print(f"✗ 错误: 上采样融合模块数量 ({num_up_fusion}) 与 brushnet_up_blocks ({num_brushnet_up}) 不匹配")
    exit(1)

print("✓ 融合模块数量匹配")

# 打印每个融合模块的通道数
print("\n[3/4] 检查融合模块通道数...")
print("\n下采样融合模块通道数:")
for i, fusion_module in enumerate(brushnet.adaptive_fusion_down):
    channels = fusion_module.channel_attention[1].in_channels
    print(f"  [{i:2d}] {channels:4d} 通道")

print("\n中间块融合模块通道数:")
channels = brushnet.adaptive_fusion_mid.channel_attention[1].in_channels
print(f"  [  ] {channels:4d} 通道")

print("\n上采样融合模块通道数:")
for i, fusion_module in enumerate(brushnet.adaptive_fusion_up):
    channels = fusion_module.channel_attention[1].in_channels
    print(f"  [{i:2d}] {channels:4d} 通道")

# 运行前向传播测试
print("\n[4/4] 运行前向传播测试...")
batch_size = 1
height, width = 64, 64

try:
    # 创建测试输入
    sample = torch.randn(batch_size, 4, height, width)
    brushnet_cond = torch.randn(batch_size, 5, height, width)
    timestep = torch.tensor([999])
    encoder_hidden_states = torch.randn(batch_size, 77, 768)

    # 前向传播
    with torch.no_grad():
        output = brushnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            brushnet_cond=brushnet_cond,
            conditioning_scale=1.0,
        )

    print("✓ 前向传播成功")
    print(f"  - 下采样输出数量: {len(output.down_block_res_samples)}")
    print(f"  - 上采样输出数量: {len(output.up_block_res_samples)}")

    # 打印每个输出的形状
    print("\n下采样输出形状:")
    for i, sample in enumerate(output.down_block_res_samples):
        print(f"  [{i:2d}] {list(sample.shape)}")

    print("\n中间块输出形状:")
    print(f"  [  ] {list(output.mid_block_res_sample.shape)}")

    print("\n上采样输出形状:")
    for i, sample in enumerate(output.up_block_res_samples):
        print(f"  [{i:2d}] {list(sample.shape)}")

except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ 所有测试通过！通道数匹配正确。")
print("=" * 80)
