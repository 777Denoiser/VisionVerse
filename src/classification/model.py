import torch
import torch.nn as nn
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self, arch='alexnet', hidden_units=512, num_classes=102):
        super(CNNModel, self).__init__()
        if arch == 'alexnet':
            self.features = models.alexnet(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, hidden_units),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(hidden_units, num_classes)
            )
        elif arch == 'vgg19':
            self.features = models.vgg19(pretrained=True).features
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, hidden_units),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(hidden_units, num_classes)
            )
        elif arch == 'resnet34':
            model = models.resnet34(pretrained=True)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Linear(512, num_classes)
        else:
            raise ValueError("Unsupported architecture")

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
