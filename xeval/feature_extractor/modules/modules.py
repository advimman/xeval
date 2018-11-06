import torch
import torch.nn as nn


class NeuralTree(nn.Module):
    def __init__(self, layers_config, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(layers_config[0], layers_config[1])
        self.linear2 = nn.Linear(layers_config[1], layers_config[2])
        self.linear3 = nn.Linear(layers_config[2], layers_config[3])
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, input):
        x = self.linear1(input)
        x = self.activation(x)
        x = self.drop(x)

        x = self.linear2(x)
        x = self.activation(x)
        x = self.drop(x)

        x = self.linear3(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, conv_channels=None):
        super().__init__()
        if conv_channels is None:
            conv_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, conv_channels, 1)
        self.norm1 = nn.BatchNorm2d(conv_channels)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=conv_channels,
                                          out_channels=conv_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding)
        self.norm2 = nn.BatchNorm2d(conv_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(conv_channels, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, padding=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
    def forward(self, x):
        return self.layer(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = ConvBnRelu(3,16,7)
        self.layer2 = ConvBnRelu(16,32,5)
        self.layer3 = ConvBnRelu(32,64,3)
        self.layer4 = ConvBnRelu(64,128,3)
        self.layer5 = ConvBnRelu(128,256,3)
        self.pool = nn.AvgPool2d(9, stride=2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
