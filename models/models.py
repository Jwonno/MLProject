import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # L2 Normalization
        return x

class MyModel(nn.Module):
    def __init__(self, embedding_dim, pretrained=True):
        super(MyModel, self).__init__()
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = models.resnet50()
                    
        self.proj_head = ProjectionHead(self.backbone.fc.weight.shape[-1], embedding_dim)
        self.backbone.fc = nn.Identity()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.proj_head(x)
        return x
    
def load_model(embedding_dim, pretrained=True):
    if pretrained:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50()
        
    model.fc = ProjectionHead(model.fc.weight.shape[-1], embedding_dim)
    
    return model