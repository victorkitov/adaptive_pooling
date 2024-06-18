import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPoolLayer(nn.Module):
    def __init__(self, pooling: str, output_size, maxpool_coef: float, avgpool_coef: float, kernel_size: int,
                 stride: int, adaptive: bool = False, t: float = 0.5,
                 concat: bool = False):
        super(CustomPoolLayer, self).__init__()
        if adaptive:
            self.maxpool = nn.AdaptiveMaxPool2d((output_size, output_size))
            self.avgpool = nn.AdaptiveAvgPool2d(output_size)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
            self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

        self.concat = concat

        maxpool_coef = torch.Tensor([maxpool_coef])
        avgpool_coef = torch.Tensor([avgpool_coef])
        self.maxpool_coef = nn.Parameter(maxpool_coef)
        self.avgpool_coef = nn.Parameter(avgpool_coef)

        coef = torch.tensor([maxpool_coef])
        self.pool_coef = nn.Parameter(coef)
        self.pooling = pooling

        self.sigmoid = nn.Sigmoid()
        self.temperature = nn.Parameter(torch.Tensor([t]))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # print("before", x.shape)
        if self.pooling == 'original':
            # print(x.shape, self.maxpool)
            out = self.maxpool(x)
        elif self.pooling == 'avg_max':
            out = self.avgpool_coef * self.avgpool(x) + self.maxpool_coef * self.maxpool(x)
        elif self.pooling == 'weighted_avg_max':
            prob_avgpooling = self.avgpool_coef ** 2 / (self.avgpool_coef ** 2 + self.maxpool_coef ** 2)
            prob_maxpooling = self.maxpool_coef ** 2 / (self.avgpool_coef ** 2 + self.maxpool_coef ** 2)
            out = prob_avgpooling * self.avgpool(x) + prob_maxpooling * self.maxpool(x)
        elif self.pooling == "sigmoid_avg_max":
            coef = self.sigmoid(self.pool_coef)
            out = (1 - coef) * self.avgpool(x) + coef * self.maxpool(x)
        elif self.pooling == "softmax_custom":
            B, C, H, W = x.shape
            unfold_x = x.unfold(2, 2, 2).reshape((B, C, -1, 4))
            w = self.softmax(unfold_x * self.temperature)
            out = torch.sum(w * unfold_x, -1).reshape((B, C, H//2, H//2))
            # print(self.temperature)
        elif self.pooling == "geometric_mean":
            coef = self.sigmoid(self.pool_coef)
            out = (self.avgpool(x) ** (1 - coef)) * (self.maxpool(x) ** coef)
            # out = torch.exp((1 - coef) * torch.log() + coef * torch.log(self.maxpool(x)))
            # print(out)
        elif self.pooling == "harmonic_mean":
            coef = self.sigmoid(self.pool_coef)
            avg_pooling = (1 - coef) / (self.avgpool(x) + 1e-6)
            max_pooling = coef / (self.maxpool(x) + 1e-6)
            out = 1 / (avg_pooling + max_pooling + 1e-6)
        elif self.pooling == "fix_adaptive":
            B, C, H, W = x.shape
            coef = self.sigmoid(self.pool_coef)
            unfold_x_22 = x.unfold(2, 2, 2).reshape((B, C, -1, 4))
            x = F.pad(torch.tensor(x), (1, 1, 1, 1), "constant", 0)
            unfold_x_44 = x.unfold(3, 4, 2).unfold(2, 4, 2).reshape((B, C, -1, 16))
            out = coef * torch.max(unfold_x_22, -1).values + (1 - coef) * torch.max(unfold_x_44, -1).values
            out = out.reshape((B, C, H//2, H//2))
        else:
            raise ValueError('Not implemented yet!')

        if self.concat:
            out = torch.cat([self.maxpool(x), self.avgpool(x)], 1)
        # print("after", out.shape)
        return out


class CNN(nn.Module):
    def __init__(self, pooling: str, num_classes: int = 10):
        super(CNN, self).__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool1 = CustomPoolLayer(pooling, 0, 0.5, 0.5, 2, 2, t=0.5, concat=False)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool2 = CustomPoolLayer(pooling, 0, 0.5, 0.5, 2, 2, t=0.5, concat=False)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # nn.MaxPool2d(kernel_size=2, stride=2),

        self.conv_layer_3 = nn.Sequential(
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = CustomPoolLayer(pooling, 0, 0.5, 0.5, 2, 2, t=0.5, concat=False)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.pool1(x)
        x = self.conv_layer_2(x)
        x = self.pool2(x)
        x = self.conv_layer_3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_layer(x)
        probas = nn.functional.softmax(logits, dim=1)
        return logits, probas
