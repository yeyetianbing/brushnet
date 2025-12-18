"""
测试带有自适应特征融合模块的BrushNet

这个脚本用于验证自适应特征融合模块的效果，并与原始BrushNet进行对比。
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import time

# 配置参数
base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"
blended = False

# 输入图像路径
image_path = "examples/brushnet/src/test_image.jpg"
mask_path = "examples/brushnet/src/test_mask.jpg"
caption = "A strawberry cake on the table."

# 条件缩放
brushnet_conditioning_scale = 1.0

print("=" * 80)
print("测试带有自适应特征融合模块的BrushNet")
print("=" * 80)

# 加载 BrushNet 和带 BrushNet 的 Stable Diffusion 管线
print("\n[1/5] 加载模型...")
start_time = time.time()

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

# 速度优化
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

load_time = time.time() - start_time
print(f"✓ 模型加载完成 (耗时: {load_time:.2f}秒)")

# 检查自适应特征融合模块是否存在
print("\n[2/5] 检查自适应特征融合模块...")
if hasattr(brushnet, 'adaptive_fusion_down'):
    print(f"✓ 检测到自适应特征融合模块!")
    print(f"  - 下采样融合模块数量: {len(brushnet.adaptive_fusion_down)}")
    print(f"  - 中间块融合模块: 1")
    print(f"  - 上采样融合模块数量: {len(brushnet.adaptive_fusion_up)}")

    # 统计参数量
    fusion_params = 0
    for module in brushnet.adaptive_fusion_down:
        fusion_params += sum(p.numel() for p in module.parameters())
    fusion_params += sum(p.numel() for p in brushnet.adaptive_fusion_mid.parameters())
    for module in brushnet.adaptive_fusion_up:
        fusion_params += sum(p.numel() for p in module.parameters())

    total_params = sum(p.numel() for p in brushnet.parameters())
    print(f"  - 融合模块参数量: {fusion_params:,} ({fusion_params/total_params*100:.2f}%)")
    print(f"  - BrushNet总参数量: {total_params:,}")
else:
    print("⚠ 警告: 未检测到自适应特征融合模块，使用原始BrushNet")

# 准备输入图像
print("\n[3/5] 准备输入图像...")
init_image = cv2.imread(image_path)[:, :, ::-1]
mask_image = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]
init_image = init_image * (1 - mask_image)

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

print(f"✓ 图像准备完成")
print(f"  - 输入图像尺寸: {init_image.size}")
print(f"  - 掩码图像尺寸: {mask_image.size}")

# 生成图像
print("\n[4/5] 生成图像...")
print(f"  - 提示词: '{caption}'")
print(f"  - 推理步数: 50")
print(f"  - 条件缩放: {brushnet_conditioning_scale}")

generator = torch.Generator("cuda").manual_seed(1234)

inference_start = time.time()
image = pipe(
    caption,
    init_image,
    mask_image,
    num_inference_steps=50,
    generator=generator,
    brushnet_conditioning_scale=brushnet_conditioning_scale
).images[0]
inference_time = time.time() - inference_start

print(f"✓ 图像生成完成 (耗时: {inference_time:.2f}秒)")
print(f"  - 平均每步耗时: {inference_time/50:.3f}秒")

# 可选的混合操作
if blended:
    print("\n[5/5] 应用混合操作...")
    image_np = np.array(image)
    init_image_np = cv2.imread(image_path)[:, :, ::-1]
    mask_np = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

    # 高斯模糊混合
    mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
    mask_blurred = mask_blurred[:, :, np.newaxis]
    mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

    image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
    image_pasted = image_pasted.astype(image_np.dtype)
    image = Image.fromarray(image_pasted)
    print("✓ 混合操作完成")
else:
    print("\n[5/5] 跳过混合操作")

# 保存结果
output_path = "output_with_adaptive_fusion.png"
image.save(output_path)

print("\n" + "=" * 80)
print("测试完成!")
print("=" * 80)
print(f"✓ 结果已保存到: {output_path}")
print(f"✓ 总耗时: {time.time() - start_time:.2f}秒")
print("\n提示: 将此结果与原始BrushNet的输出进行对比，观察质量提升")
print("=" * 80)
