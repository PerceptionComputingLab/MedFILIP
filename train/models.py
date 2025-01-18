import torch
import torch.nn as nn
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
        self.text_projection = nn.Linear(512, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts, device):
        image_feat = self.image_model(images)
        tokens = self.processor(
            text=texts,
            padding=True,
        )["input_ids"]
        tokens = torch.tensor(tokens)
        tokens = tokens.to(device)
        text_feat = self.text_model(tokens).pooler_output
        text_feat = self.text_projection(text_feat)
        image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per = logit_scale * image_feat @ text_feat.t()
        return logits_per
