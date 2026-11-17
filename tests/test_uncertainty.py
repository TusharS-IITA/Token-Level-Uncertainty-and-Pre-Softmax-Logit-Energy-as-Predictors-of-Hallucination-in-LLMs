import unittest
import torch
import numpy as np
from src.uncertainty_engine import SinglePassUncertaintyEngine
from src.data_loader import is_content_token
from src.evaluator import Evaluator

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = SinglePassUncertaintyEngine()

    def test_content_token_filter(self):
        self.assertFalse(is_content_token("the"))
        self.assertFalse(is_content_token("is"))
        self.assertFalse(is_content_token("."))
        self.assertTrue(is_content_token("quantum"))
        self.assertTrue(is_content_token("Einstein"))

    def test_token_entropies_non_negative(self):
        logits = torch.randn(2, 5, 100)
        entropies = self.engine.compute_token_entropies(logits)
        self.assertTrue(torch.all(entropies >= 0.0))

    def test_ece_computation(self):
        conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        labels = np.array([0, 0, 1, 1, 1])
        ece = Evaluator.calculate_ece(conf, labels)
        self.assertTrue(0.0 <= ece <= 1.0)

if __name__ == '__main__':
    unittest.main()
