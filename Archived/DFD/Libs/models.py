#import your nessessary libreries here
import torch
import torch.nn as nn
from torch.nn import functional as F

####    DEMO
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
#
# class DemoModel(nn.Module):
#     def __init__(self, num_classes=1):
#         super(DemoModel, self).__init__()
#         self.crop_duration = 0.47
#         self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(32 * 2590, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#        
#     def forward(self, x):
#        
#         x = x.to(next(self.parameters()).device) #!important
#        
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 2590)
#         x = F.relu(self.fc1(x))
#         x = F.sigmoid(self.fc2(x))
#         return x

class testCNN(nn.Module):
    def __init__(self):
        super(testCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 = nn.Linear(8*64*64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = x.to(next(self.parameters()).device) # to assign the input to the same device,
#         torchsummary.summary sets model to gpu but for input it does not
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        return x

