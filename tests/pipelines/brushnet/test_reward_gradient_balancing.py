import unittest

import torch

from diffusers.pipelines.brushnet.pipeline_brushnet import _balance_reward_gradients


class RewardGradientBalancingTests(unittest.TestCase):
    def setUp(self):
        self.gradients = [
            torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
            torch.tensor([[[[0.0, 0.0], [0.0, 4.0]]]]),
        ]
        self.weights = [0.5, 2.0]

    def test_zero_strength_recovers_degu_fusion(self):
        actual = _balance_reward_gradients(
            self.gradients,
            self.weights,
            scale_min=0.25,
            scale_max=4.0,
            balance_strength=0.0,
        )
        expected = sum(weight * gradient for weight, gradient in zip(self.weights, self.gradients))

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_balance_preserves_original_rms(self):
        original = sum(weight * gradient for weight, gradient in zip(self.weights, self.gradients))
        balanced = _balance_reward_gradients(
            self.gradients,
            self.weights,
            scale_min=0.25,
            scale_max=4.0,
            balance_strength=1.0,
        )

        torch.testing.assert_close(
            balanced.square().mean().sqrt(),
            original.square().mean().sqrt(),
        )
        self.assertFalse(torch.equal(balanced, original))

    def test_partial_strength_interpolates_degu_and_balanced_directions(self):
        strength = 0.25
        original = _balance_reward_gradients(
            self.gradients,
            self.weights,
            scale_min=0.25,
            scale_max=4.0,
            balance_strength=0.0,
        )
        fully_balanced = _balance_reward_gradients(
            self.gradients,
            self.weights,
            scale_min=0.25,
            scale_max=4.0,
            balance_strength=1.0,
        )
        partially_balanced = _balance_reward_gradients(
            self.gradients,
            self.weights,
            scale_min=0.25,
            scale_max=4.0,
            balance_strength=strength,
        )

        torch.testing.assert_close(partially_balanced, torch.lerp(original, fully_balanced, strength))

    def test_unit_scale_bounds_recover_degu_fusion(self):
        actual = _balance_reward_gradients(
            self.gradients,
            self.weights,
            scale_min=1.0,
            scale_max=1.0,
            balance_strength=1.0,
        )
        expected = sum(weight * gradient for weight, gradient in zip(self.weights, self.gradients))

        torch.testing.assert_close(actual, expected)

    def test_invalid_strength_raises(self):
        with self.assertRaisesRegex(ValueError, "balance_strength"):
            _balance_reward_gradients(self.gradients, self.weights, balance_strength=1.01)


if __name__ == "__main__":
    unittest.main()
