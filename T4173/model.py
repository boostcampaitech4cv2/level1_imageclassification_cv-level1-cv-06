import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('resnet34', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class resnet152_bitm(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('resnetv2_152x2_bitm', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class efficientnet_b8(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('tf_efficientnet_b8', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x

class efficientnetv2_m(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class vit_large_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('vit_large_patch16_224', pretrained = True, num_classes=num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class vit_base_224_in21k(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
class swin_large_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class swin_large_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x