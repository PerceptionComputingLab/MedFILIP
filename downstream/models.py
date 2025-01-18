import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np
import timm
from transformers import CLIPProcessor, CLIPModel


class VLM(nn.Module):
    def __init__(self, embed_dim=768, vision_model="resnet"):
        super().__init__()
        self.vision_model = vision_model
        if self.vision_model == "vit":
            self.image_model = timm.create_model(
                "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
                pretrained=True,
                num_classes=0,
            )
        elif self.vision_model == "resnet":
            self.image_model = models.resnet50(weights=True)
            num_ftrs = self.image_model.fc.in_features
            self.image_model.fc = nn.Linear(num_ftrs, embed_dim)

        self.text_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).text_model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_projection = nn.Linear(512, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts, device, embedding=False):
        image_feat = self.image_model(images)
        tokens = self.processor(
            text=texts,
            padding=True,
        )["input_ids"]
        tokens = torch.tensor(tokens)
        tokens = tokens.to(device)
        text_feat = self.text_model(tokens).pooler_output
        # text_feat = self.text_model.get_text_features(tokens)
        text_feat = self.text_projection(text_feat)
        image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per = logit_scale * image_feat @ text_feat.t()
        if embedding:
            return image_feat, text_feat
        return logits_per


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        padding=1,
        kernel_size=3,
        stride=1,
        with_nonlinearity=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        up_conv_in_channels=None,
        up_conv_out_channels=None,
        upsampling_method="conv_transpose",
    ):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2
            )
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


# This code uses a function from the pytorch-unet-resnet-50-encoder library available at https://github.com/rawmarshmellows/pytorch-unet-resnet-50-encoder/tree/master


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=1, weights=None):
        super().__init__()
        if weights:
            resnet = VLM(vision_model="resnet").image_model
            resnet.load_state_dict(weights)
        else:
            resnet = torchvision.models.resnet.resnet50(weights=None)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(
            UpBlockForUNetWithResNet50(
                in_channels=128 + 64,
                out_channels=128,
                up_conv_in_channels=256,
                up_conv_out_channels=128,
            )
        )
        up_blocks.append(
            UpBlockForUNetWithResNet50(
                in_channels=64 + 3,
                out_channels=64,
                up_conv_in_channels=128,
                up_conv_out_channels=64,
            )
        )

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
