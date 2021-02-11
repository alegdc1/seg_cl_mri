from parts_model import *


class Encoder(nn.Module):
    def __init__(self, n_channels, n_classes, no_filters):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.enc_outputs = []

        self.inc = DoubleConv(n_channels, 16)

        # TODO: create layers by using no_filters argument
        self.layer_list = nn.ModuleList([])
        for i, filters in enumerate(no_filters[:-1]):
            print("Appending filter from %d to %d" % (filters,no_filters[i+1]))
            self.layer_list.add_module("down%d" % (i+1), Down(filters, no_filters[i + 1]))
        # self.down1 = Down(16, 32)
        # self.down2 = Down(32, 64)
        # self.down3 = Down(64, 128)
        # self.down4 = Down(128, 128)
        # self.down5 = Down(128, 128)

    def forward(self, x):
        self.enc_outputs = []
        x = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x6 = self.down5(x5)

        self.enc_outputs.append(x)

        for i, module in enumerate(self.layer_list):
            out = module(x)
            if i < len(self.layer_list) - 1:
                self.enc_outputs.append(out)
            x = out

        # self.enc_outputs.append(x1)
        # self.enc_outputs.append(x2)
        # self.enc_outputs.append(x3)
        # self.enc_outputs.append(x4)
        # self.enc_outputs.append(x5)

        return x
