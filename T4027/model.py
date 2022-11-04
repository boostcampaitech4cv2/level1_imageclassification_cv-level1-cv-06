import torch.nn as nn
import torch.nn.functional as F
import timm


class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained='vggface2', num_classes=num_classes)
        for na, para in self.model.named_parameters():
            if na.startswith('last') or na.startswith('logits'):
                print(na, 'not frozen')
                continue
            else:
                para.requires_grad = False
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    
class Eff01(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
    
class ViT16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class Effb4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class SwinLarge(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class SwinBase(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class ViT384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
