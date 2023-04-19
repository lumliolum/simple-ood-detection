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
