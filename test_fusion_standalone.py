"""
独立测试改进后的自适应特征融合模块

这个脚本不依赖diffusers包，直接包含AdaptiveFeatureFusion类的代码进行测试。

使用方法：
python test_fusion_standalone.py
"""

import torch
import torch.nn as nn


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块 (Adaptive Feature Fusion Module)

    该模块通过通道注意力和空间注意力机制，动态学习特征的重要性，
    相比简单的标量缩放，能够更精细地控制条件信息的融合。

    改进版本包含：
    1. 可配置的激活函数（ReLU/SiLU/GELU等）
    2. 残差连接保留原始特征
    3. 可学习的融合强度参数

    原理：
    1. 通道注意力：学习哪些特征通道更重要（如纹理、边缘、颜色等）
    2. 空间注意力：学习哪些空间位置更重要（如掩码区域 vs 背景区域）
    3. 残差连接：保留原始特征，避免信息丢失
    4. 融合强度：控制注意力机制的影响程度

    Args:
        channels (int): 输入特征的通道数
        reduction_ratio (int): 通道注意力的降维比例，默认为16
        activation (str): 激活函数类型，可选 'relu', 'silu', 'gelu'，默认为'relu'
        use_residual (bool): 是否使用残差连接，默认为True
        fusion_strength (float): 初始融合强度，默认为0.5（0表示完全保留原始特征，1表示完全使用注意力特征）
    """
    def __init__(self, channels, reduction_ratio=16, activation='relu', use_residual=True, fusion_strength=0.5):
        super().__init__()

        self.use_residual = use_residual

        # 选择激活函数
        if activation == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation == 'silu':
            act_layer = nn.SiLU(inplace=True)
        elif activation == 'gelu':
            act_layer = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Choose from 'relu', 'silu', 'gelu'")

        # 通道注意力模块
        # 通过全局平均池化 + MLP 学习通道权重
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] -> [B, C, 1, 1]
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),  # 降维减少参数
            act_layer,
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),  # 升维恢复通道数
            nn.Sigmoid()  # 输出0-1之间的权重
        )

        # 空间注意力模块
        # 通过通道统计 + 卷积学习空间权重
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),  # 大卷积核捕获空间上下文
            nn.Sigmoid()  # 输出0-1之间的权重
        )

        # 可学习的融合强度参数
        # 使用 nn.Parameter 使其可训练，初始化为 fusion_strength
        # 通过 sigmoid 约束在 [0, 1] 范围内
        self.fusion_alpha = nn.Parameter(torch.tensor(fusion_strength))

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入特征 [B, C, H, W]

        Returns:
            torch.Tensor: 自适应融合后的特征 [B, C, H, W]
        """
        # 保存原始输入用于残差连接
        identity = x

        # 1. 通道注意力
        # 为每个通道学习一个权重，重要的通道（如掩码区域的纹理）获得更高权重
        channel_weight = self.channel_attention(x)  # [B, C, 1, 1]
        x_channel = x * channel_weight  # 广播乘法，每个通道按权重缩放

        # 2. 空间注意力
        # 计算每个空间位置的重要性统计量
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)  # [B, 1, H, W] 平均激活
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B, 1, H, W] 最大激活
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W] 拼接统计量

        spatial_weight = self.spatial_attention(spatial_input)  # [B, 1, H, W]
        x_attention = x_channel * spatial_weight  # 每个位置按权重缩放

        # 3. 融合强度控制
        # alpha 控制注意力特征的权重，(1-alpha) 控制原始特征的权重
        alpha = torch.sigmoid(self.fusion_alpha)  # 约束在 [0, 1]

        # 4. 残差连接
        if self.use_residual:
            # 加权融合：alpha * 注意力特征 + (1 - alpha) * 原始特征
            # 当 alpha=0 时，完全保留原始特征（退化为恒等映射）
            # 当 alpha=1 时，完全使用注意力特征
            output = alpha * x_attention + (1 - alpha) * identity
        else:
            # 不使用残差连接，直接使用注意力特征
            output = x_attention

        return output


def test_activation_functions():
    """测试不同激活函数的效果"""
    print("=" * 80)
    print("测试 1: 不同激活函数对比")
    print("=" * 80)

    batch_size, channels, height, width = 2, 320, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    activations = ['relu', 'silu', 'gelu']

    for act in activations:
        fusion = AdaptiveFeatureFusion(
            channels=channels,
            activation=act,
            use_residual=True,
            fusion_strength=0.5
        )
        fusion.eval()

        with torch.no_grad():
            output = fusion(x)

        # 计算输出统计信息
        mean_val = output.mean().item()
        std_val = output.std().item()
        min_val = output.min().item()
        max_val = output.max().item()

        print(f"\n激活函数: {act.upper()}")
        print(f"  输出均值: {mean_val:.6f}")
        print(f"  输出标准差: {std_val:.6f}")
        print(f"  输出范围: [{min_val:.6f}, {max_val:.6f}]")
        print(f"  参数数量: {sum(p.numel() for p in fusion.parameters())}")


def test_residual_connection():
    """测试残差连接的影响"""
    print("\n" + "=" * 80)
    print("测试 2: 残差连接对比")
    print("=" * 80)

    batch_size, channels, height, width = 2, 320, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    for use_residual in [False, True]:
        fusion = AdaptiveFeatureFusion(
            channels=channels,
            activation='relu',
            use_residual=use_residual,
            fusion_strength=0.5
        )
        fusion.eval()

        with torch.no_grad():
            output = fusion(x)

        # 计算与输入的相似度
        mse = torch.nn.functional.mse_loss(output, x).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            output.flatten(1), x.flatten(1), dim=1
        ).mean().item()

        print(f"\n残差连接: {'启用' if use_residual else '禁用'}")
        print(f"  与输入的MSE: {mse:.6f}")
        print(f"  与输入的余弦相似度: {cosine_sim:.6f}")
        print(f"  输出均值: {output.mean().item():.6f}")
        print(f"  输出标准差: {output.std().item():.6f}")


def test_fusion_strength():
    """测试不同融合强度的效果"""
    print("\n" + "=" * 80)
    print("测试 3: 融合强度参数对比")
    print("=" * 80)

    batch_size, channels, height, width = 2, 320, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    for strength in strengths:
        fusion = AdaptiveFeatureFusion(
            channels=channels,
            activation='relu',
            use_residual=True,
            fusion_strength=strength
        )
        fusion.eval()

        with torch.no_grad():
            output = fusion(x)

        # 计算与输入的相似度
        mse = torch.nn.functional.mse_loss(output, x).item()

        # 获取实际的融合强度（经过sigmoid）
        actual_alpha = torch.sigmoid(fusion.fusion_alpha).item()

        print(f"\n初始融合强度: {strength:.2f} (实际 alpha: {actual_alpha:.4f})")
        print(f"  与输入的MSE: {mse:.6f}")
        print(f"  输出均值: {output.mean().item():.6f}")
        print(f"  输出标准差: {output.std().item():.6f}")


def test_gradient_flow():
    """测试梯度流动（验证可训练性）"""
    print("\n" + "=" * 80)
    print("测试 4: 梯度流动和可训练性")
    print("=" * 80)

    batch_size, channels, height, width = 2, 320, 64, 64
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)

    fusion = AdaptiveFeatureFusion(
        channels=channels,
        activation='relu',
        use_residual=True,
        fusion_strength=0.5
    )

    # 前向传播
    output = fusion(x)
    loss = output.mean()

    # 反向传播
    loss.backward()

    # 检查梯度
    has_input_grad = x.grad is not None and x.grad.abs().sum() > 0
    has_alpha_grad = fusion.fusion_alpha.grad is not None and fusion.fusion_alpha.grad.abs() > 0

    print(f"\n输入梯度存在: {has_input_grad}")
    print(f"融合强度参数梯度存在: {has_alpha_grad}")

    if has_alpha_grad:
        print(f"  fusion_alpha 当前值: {fusion.fusion_alpha.item():.6f}")
        print(f"  fusion_alpha 梯度: {fusion.fusion_alpha.grad.item():.6f}")

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"\n可训练参数总数: {trainable_params}")

    # 列出所有可训练参数
    print("\n可训练参数列表:")
    for name, param in fusion.named_parameters():
        if param.requires_grad:
            print(f"  {name}: shape={list(param.shape)}, numel={param.numel()}")


def test_different_configurations():
    """测试不同配置组合的推荐方案"""
    print("\n" + "=" * 80)
    print("测试 5: 推荐配置方案对比")
    print("=" * 80)

    batch_size, channels, height, width = 2, 320, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    configs = [
        {
            'name': '原始配置（SiLU + 无残差）',
            'activation': 'silu',
            'use_residual': False,
            'fusion_strength': 1.0
        },
        {
            'name': '保守配置（ReLU + 残差 + 低强度）',
            'activation': 'relu',
            'use_residual': True,
            'fusion_strength': 0.3
        },
        {
            'name': '平衡配置（ReLU + 残差 + 中强度）',
            'activation': 'relu',
            'use_residual': True,
            'fusion_strength': 0.5
        },
        {
            'name': '激进配置（GELU + 残差 + 高强度）',
            'activation': 'gelu',
            'use_residual': True,
            'fusion_strength': 0.7
        }
    ]

    for config in configs:
        fusion = AdaptiveFeatureFusion(
            channels=channels,
            activation=config['activation'],
            use_residual=config['use_residual'],
            fusion_strength=config['fusion_strength']
        )
        fusion.eval()

        with torch.no_grad():
            output = fusion(x)

        # 计算指标
        mse = torch.nn.functional.mse_loss(output, x).item()
        actual_alpha = torch.sigmoid(fusion.fusion_alpha).item()

        print(f"\n{config['name']}")
        print(f"  激活函数: {config['activation']}")
        print(f"  残差连接: {'启用' if config['use_residual'] else '禁用'}")
        print(f"  融合强度: {config['fusion_strength']} (实际: {actual_alpha:.4f})")
        print(f"  与输入MSE: {mse:.6f}")
        print(f"  输出均值: {output.mean().item():.6f}")
        print(f"  输出标准差: {output.std().item():.6f}")


def main():
    print("\n" + "=" * 80)
    print("自适应特征融合模块改进测试")
    print("=" * 80)
    print("\n改进内容:")
    print("1. 可配置的激活函数（ReLU/SiLU/GELU）")
    print("2. 残差连接保留原始特征")
    print("3. 可学习的融合强度参数")
    print()

    # 运行所有测试
    test_activation_functions()
    test_residual_connection()
    test_fusion_strength()
    test_gradient_flow()
    test_different_configurations()

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n推荐使用方案:")
    print("1. 如果像素级指标很重要（PSNR, MSE, LPIPS）:")
    print("   - 使用 ReLU 激活函数")
    print("   - 启用残差连接 (use_residual=True)")
    print("   - 设置较低的融合强度 (fusion_strength=0.3)")
    print()
    print("2. 如果需要平衡像素级和感知质量:")
    print("   - 使用 ReLU 或 GELU 激活函数")
    print("   - 启用残差连接 (use_residual=True)")
    print("   - 设置中等融合强度 (fusion_strength=0.5)")
    print()
    print("3. 训练时建议:")
    print("   - 让 fusion_alpha 参数可学习，模型会自动调整最优强度")
    print("   - 可以尝试不同激活函数，观察验证集指标")
    print("=" * 80)


if __name__ == "__main__":
    main()
