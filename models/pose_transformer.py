import torch
import torch.nn as nn
from models.resnet import ResNetBackbone
from models.transformer import TransformerEncoder, TransformerDecoder

class PoseTransformer(nn.Module):
    def __init__(self, num_keypoints=17, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, pretrained=True):
        super(PoseTransformer, self).__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_encoder_layers)
        self.transformer_decoder = TransformerDecoder(num_keypoints, d_model, nhead, num_decoder_layers)

    def forward(self, x):
        # Extract ResNet features
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, num_frames, -1, features.size(-2), features.size(-1))
        features = features.permute(1, 0, 3, 4, 2).contiguous()
        # Adjust the dimension order to match the Transformer input

        # Flatten features into 2D
        features = features.view(num_frames, batch_size, -1)

        # Transformer encoding
        memory = self.transformer_encoder(features)

        # Transformer decoding
        output = self.transformer_decoder(memory)
        return output

if __name__ == '__main__':
    # Test if the PoseTransformer model is working correcting
    model = PoseTransformer(pretrained=False)
    # Assume the input consists of 2 samples, each sample with 10 frames, and each frame sized 3 * 224 * 224
    input_tensor = torch.randn(2, 10, 3, 224, 224)
    output = model(input_tensor)
    # It should output [17, 2, 2], representing 17 keypoints, 2 samples, and each keypoint with 2 coordinates
    print("Output shape:", output.shape)