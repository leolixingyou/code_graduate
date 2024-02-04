import torch
import torch.nn as nn
import numpy as np


class ErrorPredictionNN_5(nn.Module):

    ##### 5layers
    def __init__(self):
        super(ErrorPredictionNN_5, self).__init__()

        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

class ErrorPredictionNN_2(nn.Module):
    ##### 2layers
    def __init__(self):
        super(ErrorPredictionNN_2, self).__init__()

        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class ErrorPredictionNN_3(nn.Module):
    ##### 3layers
    def __init__(self):
        super(ErrorPredictionNN_3, self).__init__()

        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class ErrorPredictionNN_10(nn.Module):
    ##### 10layers
    def __init__(self):
        super(ErrorPredictionNN_10, self).__init__()

        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.Linear(128, 256)
        self.layer5 = nn.Linear(256, 512)
        self.layer6 = nn.Linear(512, 256)
        self.layer7 = nn.Linear(256, 128)
        self.layer8 = nn.Linear(128, 64)
        self.layer9 = nn.Linear(64, 32)
        self.layer10 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = torch.relu(self.layer7(x))
        x = torch.relu(self.layer8(x))
        x = torch.relu(self.layer9(x))
        x = self.layer10(x)
        return x

class ErrorPredictionNN_15(nn.Module):
    ##### 15layers
    def __init__(self):
        super(ErrorPredictionNN_15, self).__init__()

        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.Linear(128, 256)
        self.layer5 = nn.Linear(256, 512)
        self.layer6 = nn.Linear(512, 1024)
        self.layer7 = nn.Linear(1024, 2048)
        self.layer8 = nn.Linear(2048, 4096)
        self.layer9 = nn.Linear(4096, 2048)
        self.layer10 = nn.Linear(2048, 1024)
        self.layer11 = nn.Linear(1024, 512)
        self.layer12 = nn.Linear(512, 256)
        self.layer13 = nn.Linear(256, 128)
        self.layer14 = nn.Linear(128, 64)
        self.layer15 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = torch.relu(self.layer7(x))
        x = torch.relu(self.layer8(x))
        x = torch.relu(self.layer9(x))
        x = torch.relu(self.layer10(x))
        x = torch.relu(self.layer11(x))
        x = torch.relu(self.layer12(x))
        x = torch.relu(self.layer13(x))
        x = torch.relu(self.layer14(x))
        x = self.layer15(x)
        return x

class ErrorPredictionNN_20(nn.Module):
    ##### 20layers
    def __init__(self):
        super(ErrorPredictionNN_20, self).__init__()

        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.Linear(128, 256)
        self.layer5 = nn.Linear(256, 512)
        self.layer6 = nn.Linear(512, 1024)
        self.layer7 = nn.Linear(1024, 2048)
        self.layer8 = nn.Linear(2048, 4096)
        self.layer9 = nn.Linear(4096, 8192)
        self.layer10 = nn.Linear(8192, 16384)
        self.layer11 = nn.Linear(16384, 8192)
        self.layer12 = nn.Linear(8192, 4096)
        self.layer13 = nn.Linear(4096, 2048)
        self.layer14 = nn.Linear(2048, 1024)
        self.layer15 = nn.Linear(1024, 512)
        self.layer16 = nn.Linear(512, 256)
        self.layer17 = nn.Linear(256, 128)
        self.layer18 = nn.Linear(128, 64)
        self.layer19 = nn.Linear(64, 32)
        self.layer20 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = torch.relu(self.layer7(x))
        x = torch.relu(self.layer8(x))
        x = torch.relu(self.layer9(x))
        x = torch.relu(self.layer10(x))
        x = torch.relu(self.layer11(x))
        x = torch.relu(self.layer12(x))
        x = torch.relu(self.layer13(x))
        x = torch.relu(self.layer14(x))
        x = torch.relu(self.layer15(x))
        x = torch.relu(self.layer16(x))
        x = torch.relu(self.layer17(x))
        x = torch.relu(self.layer18(x))
        x = torch.relu(self.layer19(x))
        x = self.layer20(x)
        return x

def judge_layers(num_layers):
    if num_layers == 2:
        return ErrorPredictionNN_2()

    if num_layers == 3:
        return ErrorPredictionNN_3()

    if num_layers == 5:
        return ErrorPredictionNN_5()

    if num_layers == 10:
        return ErrorPredictionNN_10()

    if num_layers == 15:
        return ErrorPredictionNN_15()

    if num_layers == 20:
        return ErrorPredictionNN_20()
