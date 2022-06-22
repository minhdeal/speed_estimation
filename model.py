from torch import nn


class SpeedCNN(nn.Module):
    def __init__(self):
        super(SpeedCNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2))
        self.bn_1 = nn.BatchNorm2d(24)
        self.conv_2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2))
        self.bn_2 = nn.BatchNorm2d(36)
        self.conv_3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2))
        self.bn_3 = nn.BatchNorm2d(48)
        self.conv_4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.bn_4 = nn.BatchNorm2d(64)
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='valid')
        self.fc1 = nn.Linear(65664, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.CELU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.activation(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activation(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.activation(x)

        x = self.conv_5(x)
        x = self.flatten(x)
        x = self.activation(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)

        return x
