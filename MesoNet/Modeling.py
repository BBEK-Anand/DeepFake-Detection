import torch
import torch.nn as nn
import torch.nn.functional as F


# Meso4 Model
class Meso4_01(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_01, self).__init__()
        self.input_shape = (3,128,128)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        self.fc1 = nn.Linear(16 * 4 * 4, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Meso4 Model
class Meso4_02(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_02, self).__init__()
        self.input_shape = (3,128,128)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Meso4 Model
class Meso4_03(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_03, self).__init__()
        self.input_shape = (3,128,128)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4,stride=3,padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=8,stride=3,padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=6,stride=3,padding=2)
        
        self.fc1 = nn.Linear(16 * 4*4, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)  # Flatten feature maps

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Meso4 Model
class Meso4_04(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_04, self).__init__()
        self.input_shape = (3,150,150)
        self.conv1 = nn.Conv2d(3, 10, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(20,20, kernel_size=7, padding=3)  

        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(10)
        self.bn3 = nn.BatchNorm2d(20)
        self.bn4 = nn.BatchNorm2d(20)

        self.pool = nn.MaxPool2d(kernel_size=3,stride=3,padding=1)
        
        self.fc1 = nn.Linear(20 * 2*2, 20)
        self.fc2 = nn.Linear(20, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Meso4 Model
class Meso4_05(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_05, self).__init__()
        self.input_shape = (3,150,150)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        
        self.fc1 = nn.Linear(16 * 9 * 9, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x










# Meso4 Model
class Meso4_06(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_06, self).__init__()
        self.input_shape = (3,150,150)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=3,stride=3,padding=0)
        
        self.fc1 = nn.Linear(16 * 5*5, 20)
        self.fc2 = nn.Linear(20, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Meso4 Model
class Meso4_07(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_07, self).__init__()
        self.input_shape = (3,180,180)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=3,stride=3,padding=1)
        
        self.fc1 = nn.Linear(16 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Meso4 Model
class Meso4_08(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_08, self).__init__()
        self.input_shape = (3,240,240)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        
        self.fc1 = nn.Linear(16 * 15 * 15, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Meso4 Model
class Meso4_09(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4_09, self).__init__()
        self.input_shape = (3,240,240)
        self.conv1 = nn.Conv2d(3, 15, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(15, 15, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(15, 24, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(24, 24, kernel_size=5, padding=2)  

        self.bn1 = nn.BatchNorm2d(15)
        self.bn2 = nn.BatchNorm2d(15)
        self.bn3 = nn.BatchNorm2d(24)
        self.bn4 = nn.BatchNorm2d(24)

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=4,stride=4,padding=0)
        
        self.fc1 = nn.Linear(24 * 7 * 7, 15)
        self.fc2 = nn.Linear(15, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# # Detect if GPU is available and use it
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1 = Meso4_09(num_classes=1).to(device)  # Move model to GPU if available
# summary(model1,input_size=model1.input_shape)