from typing import Tuple

import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import InterpolationMode


def get_train_transforms(
    crop_size: int,
    hflip_prob: float = 0.5,
    random_erase_prob: float = 0.1,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    interpolation=InterpolationMode.BILINEAR
):
    transform = [
        torchvision.transforms.RandomResizedCrop(size=crop_size)
    ]
    if hflip_prob > 0:
        transform.append(torchvision.transforms.RandomHorizontalFlip(hflip_prob)) # type: ignore

    transform.extend(
        [
            torchvision.transforms.TrivialAugmentWide(interpolation=interpolation),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ] # type: ignore
    )

    if random_erase_prob > 0:
        transform.append(torchvision.transforms.RandomErasing(random_erase_prob)) # type: ignore

    transform = torchvision.transforms.Compose(transform)

    return transform


def get_test_transforms(
    resize_size: int,
    crop_size: int,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    interpolation=InterpolationMode.BILINEAR
):
    transform = [
        torchvision.transforms.Resize(resize_size, interpolation=interpolation),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ]
    transform = torchvision.transforms.Compose(transform)

    return transform


# copied from here : https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
class RandomCutMix(nn.Module):
    """
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0):
        super(RandomCutMix, self).__init__()

        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")

        if alpha <= 0:
            raise ValueError("Alpha param has to be positive number")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha

    def forward(self, batch: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        batch = batch.clone()
        target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        # if sampled number (from U(0, 1)) is greater than equal to self.p,
        # then original batch will be returned without any transformation.
        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        H, W = batch.shape[-2], batch.shape[-1]

        # sample r_x and r_y. which are center of the bounding box.
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * np.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target
