from parts_model import *
import numpy as np

class Decoder(nn.Module):
    def __init__(self, config, encoder):
        super(Decoder, self).__init__()
        self.n_channels = config['n_channels']
        self.n_classes = config['n_classes']
        self.no_filters = config['no_filters']
        self.n_dec_blocks = config['n_dec_blocks']
        self.encoder = encoder
        self.interp_method = "bilinear" if config["interp_method"] is None else config["interp_method"]
        self.factor = 2 if config["factor"] is None else config["factor"]
        self.enc_outputs = encoder.enc_outputs

        self.layer_list = nn.ModuleList([])
        self.filters_reversed = self.no_filters[::-1]
        for i, filters in enumerate(self.filters_reversed):
            if i + 1 > self.n_dec_blocks:
                break
            # print("Appending filter from %d to %d" % (filters,self.filters_reversed[i+1]))
            self.layer_list.add_module("up%d" % (i + 1), Up(filters, self.filters_reversed[i+1], self.interp_method))
        # self.up1 = Up(128, 256 // self.factor, self.interp_method)
        # self.up2 = Up(128, 256 // self.factor, self.interp_method)
        # self.up3 = Up(128, 128 // self.factor, self.interp_method)
        # self.up4 = Up(64, 64 // self.factor, self.interp_method)
        # self.up5 = Up(32, 32 // self.factor, self.interp_method)

    def forward(self, x):
        for i, module in enumerate (self.layer_list):
            out = module(x, self.encoder.enc_outputs[-(i+1)])
            x = out


        # if self.n_dec_blocks >= 1:
        #     x = self.up1(x, self.enc_outputs[4])
        #     if self.n_dec_blocks >= 2:
        #         x = self.up2(x, self.enc_outputs[3])
        #         if self.n_dec_blocks >= 3:
        #             x = self.up3(x, self.enc_outputs[2])
        #             if self.n_dec_blocks >= 4:
        #                 x = self.up4(x, self.enc_outputs[1])
        #                 if self.n_dec_blocks >= 5:
        #                     x = self.up5(x, self.enc_outputs[0])
        return x