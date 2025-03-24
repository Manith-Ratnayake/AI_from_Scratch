import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F



VGG16 = {
        "VGG11" : [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"],
        "VGG13" : [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        "VGG16" : [64, 64, "M", 128, 128, "M", 256, 256, 256, ]
        }

class VGG(nn.Module):

    def __init__(self, in_channels, num_classes = 1000):
        super(VGG, self).__init__()
        self.in_channel  = in_channels
        self.conv_layers = create_conv_layers(VGG) 
        self.fcs         = nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes)
                ) 

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channel = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channel = x 

                layer += [nn.Conv2d(in_channels = in_channels, out_channel = out_channel, kernel_size = (3,3), stride=(1,1) ,\
                                    padding=(1,1)), nn.BatchNorm2d(x), nn.ReLU()] #batchNorm wasnt introduced

                in_channels = x 
            
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

            return nn.Sequential(*layers)


model = VGG(in_channels=3, num_classes=1000)
x     = torch.randn(1,3,244,244)
print(model(x).shape)



