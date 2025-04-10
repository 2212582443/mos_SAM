import torch, random


class RandomScaleIntensity(torch.nn.Module):
    def __init__(self, scale_lower: float = 0.9, scale_upper: float = 1.1, p=0.2):
        super().__init__()
        self.sacle_lower = scale_lower
        self.scale_upper = scale_upper
        self.scale = scale_upper - scale_lower
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: float, 0~1
        """
        if random.random() >= self.p:
            return x

        scale = random.uniform(self.sacle_lower, self.scale_upper)
        return (x * scale).clamp(0, 1)
