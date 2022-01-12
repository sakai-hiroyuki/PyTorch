from torch import nn

__all__ = ['SimpleCNN_MNIST']


class SimpleCNN_MNIST(nn.Module):
    def __init__(self):
        super(SimpleCNN_MNIST, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout(p=0.25),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1024, out_features=200),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=200, out_features=10),
        )

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out
