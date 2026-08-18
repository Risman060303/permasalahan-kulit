
import torch
import torchvision
from torch import nn

def efficientnetb0(OUT_FEATURES: int):

  weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT
  model = torchvision.models.efficientnet_b0(weights=weight)

  # (classifier): Sequential(
  #   (0): Dropout(p=0.2, inplace=True)
  #   (1): Linear(in_features=1280, out_features=1000, bias=True)
  # )

  for param in model.parameters():
    param.requires_grad = False

  model.classifier = nn.Sequential(
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1280,
                out_features=OUT_FEATURES,
                bias=True)
  )

  model.name = "EffNetB0"
  print(f"[INFO] {model.name} telah berhasil dibuat")

  return model
