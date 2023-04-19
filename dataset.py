from PIL import Image
from typing import Dict, List, Callable
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
            self,
            image_paths: List[str],
            labels: List,
            transforms: Callable,
            label2idx: Dict[str, int],
        ) -> None:

        if len(image_paths) != len(labels):
            raise ValueError(f"Image paths and labels should be of same length, but received {len(image_paths)}, {len(labels)}")

        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.label2idx = label2idx

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict:
        path = self.image_paths[index]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)

        label = self.label2idx[self.labels[index]]
        return {"image": img, "label": label}


class ImageTestDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transforms: Callable,
    ) -> None:
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict:
        path = self.image_paths[index]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return {"image": img}
