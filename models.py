import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet, ResNet50_Weights


class OODResNetModel(nn.Module):
    def __init__(self, backbone: ResNet, num_classes: int) -> None:
        super(OODResNetModel, self).__init__()
        self.backbone = backbone
        if not isinstance(self.backbone, ResNet):
            raise ValueError(f"OOD Resnet model expects backbone of type Resnet, but received = {type(self.backbone)}")

        self.layer_names = ["layer1", "layer2", "layer3", "layer4"]

        # replace the fc layer of the backbone.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        embds = []

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # to understand what is happening here, you should go through
        # this block : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L273-L276
        # carefully.
        for layername in self.layer_names:
            layer = getattr(self.backbone, layername)
            for idx in range(len(layer)):
                # pass to the layer 1 first block
                x = layer[idx](x)
                # the output x is actually just after the residual block.
                # x is of shape (B, C, H, W) so we will take mean on 2, 3 dim
                # store it in the embds list.
                embds.append(torch.mean(x, dim=(2, 3)))

        # this will return output shape of (B, C, 1, 1)
        x = self.backbone.avgpool(x)
        # here x got converted to (B, C)
        x = torch.flatten(x, 1)
        # here note that we are not adding x just before fc to embds
        # the reason that this is taken care at the end of fourth layer.
        # because the avgpool that Resnet uses is global average pooling.
        x = self.backbone.fc(x)
        return x, embds


if __name__ == "__main__":
    backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = OODResNetModel(backbone, num_classes=3)

    x = torch.randn((4, 3, 32, 32))
    out, embds = model(x)
    import pdb; pdb.set_trace()
