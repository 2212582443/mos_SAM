from dataclasses import dataclass


@dataclass
class ImageShape(object):
    width: int = 224
    height: int = 224
    deep: int = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def dim(self):
        if self.deep > 0:
            return 3
        return 2

    def is_3d(self):
        return self.deep > 0


