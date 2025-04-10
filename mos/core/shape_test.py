import unittest

from .shape import ImageShape


class ImageShapeTest(unittest.TestCase):
    def test_2d_shape(self):
        shape = ImageShape()
        self.assertEqual(shape.dim(), 2)
        self.assertFalse(shape.is_3d())
        self.assertEqual(shape.width, 224)
        self.assertEqual(shape.height, 224)
