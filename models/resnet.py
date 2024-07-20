import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        resnet = resnet50(pretrained=pretrained)
        # Remove the last two layers of ResNet, the fully connected layer and the pooling layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == '__main__':
    # Test if ResNetBackbone is working correctly
    model = ResNetBackbone(pretrained=False)
    input_tensor = torch.rand(1, 3, 224, 224)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)