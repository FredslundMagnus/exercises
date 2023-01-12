import torch
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

