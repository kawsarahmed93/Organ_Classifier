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

if __name__ == '__main__':
    imgs = torch.zeros((2,3,224,224)).to("cuda:0")  
    model = DenseNet121().to("cuda:0")
    out = model(imgs)