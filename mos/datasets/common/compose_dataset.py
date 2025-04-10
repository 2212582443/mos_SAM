
from typing import List
from torch.utils.data import Dataset


class ComposeDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError(f"index out of range: {index}")

    def get_item_info(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset.get_item_info(index)
            index -= len(dataset)
        raise IndexError(f"index out of range: {index}")
