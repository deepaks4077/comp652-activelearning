import torch
import torch.nn as nn

class convnet_mnist(nn.Module):
    def __init__(self, num_classes = 10, input_size = 28):
        super(convnet_mnist, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        test_inp = torch.randn((1,1,self.input_size,self.input_size))

        out = self.layer1(test_inp)
        out = self.layer2(out)

        self.fc = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(out.numel(), self.num_classes)
                )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
