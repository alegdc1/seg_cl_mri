from encoder import Encoder
import torch.nn as nn
import pdb

class EncoderPretrainNet(nn.Module):
    def __init__(self, n_channels, n_classes, no_filters):
        super().__init__()

        self.enc = Encoder(n_channels, n_classes, no_filters)
        self.g1= nn.Sequential(nn.Flatten(),
                                nn.Linear(3200, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 128)
                                )

    def forward(self, x):
        x = self.enc(x)
        #pdb.set_trace()
        x = self.g1(x)

        return x

