import unittest

import torch

from har.pooling import AttentionPool, AvgMaxPool, GeM


class PoolingTests(unittest.TestCase):
    def test_attention_initially_matches_mean(self):
        tensor = torch.rand(2, 512, 3, 4)
        self.assertTrue(torch.allclose(AttentionPool()(tensor).flatten(1), tensor.mean((-2, -1)), atol=1e-6))

    def test_gem_preserves_constant_and_learns_exponent(self):
        pool = GeM()
        self.assertTrue(torch.allclose(pool(torch.ones(2, 4, 3, 3)), torch.ones(2, 4, 1, 1)))
        output = pool(torch.rand(2, 4, 3, 3))
        output.sum().backward()
        self.assertIsNotNone(pool.raw_p.grad)
        self.assertTrue(torch.isfinite(pool.raw_p.grad))
        self.assertGreater(pool.raw_p.grad.abs().item(), 0)

    def test_avgmax_combines_statistics(self):
        tensor = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
        output = AvgMaxPool()(tensor)
        self.assertEqual(tuple(output.shape), (1, 2, 1, 1))
        self.assertEqual(output.flatten().tolist(), [2.5, 4.0])
        output.sum().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
