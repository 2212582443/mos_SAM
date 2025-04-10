import unittest

from utils.tensors import load_tensor_file


class TensorTest(unittest.TestCase):
    def test_load_tensor_file_npz(self):
        file = "tests/resource/sample.npz"
        data = load_tensor_file(file)
        self.assertIsNotNone(data['image'])
        self.assertIsNotNone(data['segment'])

    def test_load_tensor_file_npy(self):
        file = "tests/resource/sample.npy"
        data = load_tensor_file(file)
        self.assertIsNotNone(data[''])
