import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast

class DenseNet121(nn.Module):
    def __init__(self, num_classes: int=20):
        super().__init__()
        # get densenet-121 architecture
        backbone_model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        modules = list(backbone_model.children())[0]

        # stem
        self.stem = nn.Sequential(*modules[:4])
        
        # dense block 1
        self.db1 = nn.Sequential(*modules[4:5])
        self.trn1 = nn.Sequential(*list(modules[5])[:-1])

        # dense block 2
        self.db2 = nn.Sequential(*modules[6:7])
        self.trn2 = nn.Sequential(*list(modules[7])[:-1])

        # dense block 3
        self.db3 = nn.Sequential(*modules[8:9])
        self.trn3 = nn.Sequential(*list(modules[9])[:-1])

        # dense block 4
        self.db4 = nn.Sequential(*modules[10:])

        # avg layer for applying after dense block-1,2,3
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # final fc
        self.classifier = nn.Linear(1024, num_classes)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()
        
    @autocast()
    def forward(self, x):
        x = self.stem(x)

        x = self.db1(x)
        x1 = self.trn1(x)
        x = self.avg(x1)

        x = self.db2(x)
        x2 = self.trn2(x)
        x = self.avg(x2)

        x = self.db3(x)
        x3 = self.trn3(x)
        x = self.avg(x3)

        x = self.db4(x)
        xg = F.relu(x)

        xg_pool = F.adaptive_avg_pool2d(xg, (1,1)).flatten(1)       
        logits = self.classifier(xg_pool)
         
        return {
            'logits': logits,
            }

class ConvNeXt_Small(nn.Module):
    def __init__(self, num_classes: int=20):
        super().__init__()
        model = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1)                                    
        self.layer1 = nn.Sequential(
                            model.features[0],            
                            model.features[1],
                            )
        self.layer2 = nn.Sequential(
                            model.features[2],            
                            model.features[3],
                            )
        self.layer3 = nn.Sequential(
                            model.features[4],            
                            model.features[5],
                            )
        self.layer4 = nn.Sequential(
                            model.features[6],            
                            model.features[7],
                            )

        self.classifier = model.classifier 
        self.classifier[2] = nn.Linear(768, num_classes)
        self.classifier[2].weight.data.normal_(0, 0.01)
        self.classifier[2].bias.data.zero_()

    @autocast()
    def forward(self, x:torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        xg = self.layer4(x)
        xg_pool = F.adaptive_avg_pool2d(xg, (1,1))
        logits = self.classifier(xg_pool)
        
        return {
            'logits': logits,
            }
    

class ConvNeXt_Large(nn.Module):
    def __init__(self, num_classes: int = 20, pretrained: bool = True):
        super().__init__()

        weights = (
            torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        model = torchvision.models.convnext_large(weights=weights)

        # Keep the same staged feature blocks layout as your small version
        self.layer1 = nn.Sequential(model.features[0], model.features[1])
        self.layer2 = nn.Sequential(model.features[2], model.features[3])
        self.layer3 = nn.Sequential(model.features[4], model.features[5])
        self.layer4 = nn.Sequential(model.features[6], model.features[7])

        # Replace final linear layer for your num_classes
        self.classifier = model.classifier

        # --- robust way: read in_features from the existing Linear ---
        in_features = self.classifier[2].in_features
        self.classifier[2] = nn.Linear(in_features, num_classes)

        # init like you did
        nn.init.normal_(self.classifier[2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier[2].bias)

    @autocast()
    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        xg = self.layer4(x)

        xg_pool = F.adaptive_avg_pool2d(xg, (1, 1))
        logits = self.classifier(xg_pool)

        return {"logits": logits}


if __name__ == "__main__":
    imgs = torch.zeros((2, 3, 224, 224))
    model = ConvNeXt_Large(num_classes=20, pretrained=False)
    out = model(imgs)
    print(out["logits"].shape)  # torch.Size([2, 20])
