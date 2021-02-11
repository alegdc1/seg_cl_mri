from parts_model import *
import numpy as np
from encoder import Encoder
from decoder import Decoder


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, interp_method="bilinear"):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.interp_method = interp_method

        no_filters = np.array([16, 32, 64, 128, 128, 128])

        # TODO: give correct and all arguments
        self.encoder = Encoder(n_channels, n_classes, no_filters)
        self.decoder = Decoder(n_channels, n_classes, no_filters, self.encoder, interp_method=interp_method)
        self.out_conv = OutConv(no_filters[0], n_classes)

    def forward(self, x):
        x = self.encoder(x)
        self.decoder.enc_outputs = self.encoder.enc_outputs

        pdb.set_trace()

        x = self.decoder(x)
        pdb.set_trace()
        logits = self.out_conv(x)
        pdb.set_trace()
        return logits
