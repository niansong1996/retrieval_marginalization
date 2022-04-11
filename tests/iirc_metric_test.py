import unittest
import torch

from numpy.testing import assert_allclose
from rranm_modules.utils.iirc_metric import SetFMeasure


class TestSetFMeasure(unittest.TestCase):
    def test_basic(self):
        set_f_measure = SetFMeasure()
        predict = torch.Tensor([[0, 1, 1, 0], [1, 0, 0, 0]])
        gold_lb = torch.Tensor([[0, 0, 1, 0], [1, 0, 1, 1]])

        set_f_measure(predict, gold_lb)

        p, r, f = set_f_measure.get_metric()

        assert_allclose(p, 0.75, atol=1e-3)
        assert_allclose(r, 0.6666, atol=1e-3)
        assert_allclose(f, 0.7059, atol=1e-3)

    def test_empty_set(self):
        set_f_measure = SetFMeasure()
        predict = torch.Tensor([[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]])
        gold_lb = torch.Tensor([[1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        set_f_measure(predict, gold_lb)

        p, r, f = set_f_measure.get_metric()

        assert_allclose(p, 0.6666, atol=1e-3)
        assert_allclose(r, 0.5, atol=1e-3)
        assert_allclose(f, 0.5714, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
