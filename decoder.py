from parts_model import *

class Decoder(nn.Module):
    def __init__(self, n_channels, n_classes, no_filters, encoder, interp_method='bilinear', factor=2):
        super(Decoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.enc_outputs = encoder.enc_outputs

        # TODO: create layers by using no_filters argument

        self.up1 = Up(128, 256 // factor, interp_method)
        self.up2 = Up(128, 256 // factor, interp_method)
        self.up3 = Up(128, 128 // factor, interp_method)
        self.up4 = Up(64, 64 // factor, interp_method)
        self.up5 = Up(32, 32 // factor, interp_method)

    def forward(self, x):
        x = self.up1(x, self.enc_outputs[4])
        x = self.up2(x, self.enc_outputs[3])
        x = self.up3(x, self.enc_outputs[2])
        x = self.up4(x, self.enc_outputs[1])
        x = self.up5(x, self.enc_outputs[0])
        return x