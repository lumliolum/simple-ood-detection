from torch.utils.data.dataloader import default_collate

from transforms import RandomCutMix


class Collator:
    def __init__(self, num_classes: int, cutmix_alpha: float):
        self.cutmix_alpha = cutmix_alpha

        if self.cutmix_alpha > 0:
            self.cutmix = RandomCutMix(num_classes=num_classes, p=1.0, alpha=self.cutmix_alpha)
        else:
            self.cutmix = None

    def collate_fn(self, inputs):
        """
            inputs will be a list where each element is output of
            dataset.__getitem__[index].
        """
        # apply the default collate function
        inputs = default_collate(inputs)
        if self.cutmix is not None:
            # apply cutmix on the batch.
            batch, target = self.cutmix(inputs["image"], inputs["label"])
            inputs = {
                "image": batch.float(),
                "label": target.float(),
            }
        else:
            inputs = {
                "image": inputs["image"].float(),
                "label": inputs["label"].long(),
            }

        return inputs
