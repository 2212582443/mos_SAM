import unittest

from utils.files import get_file_ext


class FileUtilsTest(unittest.TestCase):
    def test_get_file_ext(self):
        file = "abd.e.nii.gz"
        ext = get_file_ext(file)
        self.assertEqual(ext, ".nii.gz")

        file = "abc.npz"
        ext = get_file_ext(file)
        self.assertEqual(ext, ".npz")
