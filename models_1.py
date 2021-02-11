import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class modelfactory:
    def __init__(self,cfg,override_num_classes=0):
        self.img_size_x=cfg.img_size_x
        self.img_size_y=cfg.img_size_y
        self.num_classes=cfg.num_classes
        self.num_channels=cfg.num_channels

        self.interp_val = cfg.interp_val
        self.img_size_flat=cfg.img_size_flat
        self.batch_size=cfg.batch_size_ft

        self.mtask_bs=cfg.mtask_bs

        if(override_num_classes==1):
            self.num_classes=2

    def cos_sim(self, vec_a, vec_b, temp_fac):

        cos_sim_val = nn.CosineSimilarity(dim=1, eps=1e-6)/temp_fac

        return cos_sim_val(vec_a, vec_b)

    def encoder_network(self, in_channels, out_channels, kernel_size, padding, num_layers=5, encoder_list_return=0):
        encoder_inter_net = nn.Sequential()
        for i in range(num_layers):
            encoder_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
            encoder_inter_net.append(encoder_block)
        final_block =  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        encoder_net = nn.Sequential(encoder_inter_net, final_block)
        if(encoder_list_return==1):
            return encoder_net, encoder_inter_net
        else:
            return encoder_net

    def encoder_pretrain_net(self, in_channels, out_channels, kernel_size, padding, num_layers=5, encoder_list_return=0):
        encoder_net = self.encoder_network(in_channels, out_channels, kernel_size, padding, num_layers=5, encoder_list_return=0)
        reg_flat = torch.flatten(encoder_net)

        reg_NN_1= nn.Sequential(
        nn.Linear(reg_flat, 1024),
        torch.nn.ReLU()
        )
        reg_pred = nn.Linear(reg_NN_1, 128)

        net_global_loss = 0

        if (global_loss_exp_no == 1):
            ######################
            # G^{D-} - Proposed variant
            # We split each volume into n_parts and select 1 image from each n_part of the volume
            # We select the negative samples that we want to contrast against for a given positive image.
            # Example: if positive image is from partition 1 of volume 1, then NO negative sample are taken from partition 1 of any other volume (including volume 1).
            ######################
            bs = 3 * self.batch_size
            # loop over each pair of positive images in the batch to calculate the Net global contrastive loss over the whole batch.
            for pos_index in range(0, self.batch_size, 1):
                # indexes of positive pair of samples (x_1,x_2,x_3) - we can make 3 pairs: (x_1,x_2), (x_1,x_3), (x_2,x_3)
                num_i1 = np.arange(pos_index, pos_index + 1, dtype=np.int32)
                j = self.batch_size + pos_index
                num_i2 = np.arange(j, j + 1, dtype=np.int32)
                j = 2 * self.batch_size + pos_index
                num_i3 = np.arange(j, j + 1, dtype=np.int32)
                # print('n1,n2,n3',num_i1,num_i2,num_i3)

                # indexes of corresponding negative samples as per positive pair of samples: (x_1,x_2), (x_1,x_3), (x_2,x_3)
                den_index_net = np.arange(0, bs, dtype=np.int32)

                # Pruning the negative samples
                # Deleting the indexes of the samples in the batch used as negative samples for a given positive image. These indexes belong to identical partitions in other volumes in the batch.
                # Example: if positive image is from partition 1 of volume 1, then NO negative sample are taken from partition 1 of any other volume (including volume 1) in the batch
                ind_l = []
                rem = int(num_i1) % n_parts
                for not_neg_index in range(rem, bs, 4):
                    ind_l.append(not_neg_index)

                # print('ind_l',ind_l)
                den_indexes = np.delete(den_index_net, ind_l)
                # print('d1',den_i1,len(den_i1))

                # gather required positive samples x_1,x_2,x_3 for the numerator term
                x_num_i1 = nn.gather(reg_pred, num_i1)
                x_num_i2 = nn.gather(reg_pred, num_i2)
                x_num_i3 = nn.gather(reg_pred, num_i3)

                # gather required negative samples x_1,x_2,x_3 for the denominator term
                x_den = nn.gather(reg_pred, den_indexes)

                # calculate cosine similarity score + global contrastive loss for each pair of positive images

                # for positive pair (x_1,x_2);
                # numerator of loss term (num_i1_i2_ss) & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
                num_i1_i2_ss = self.cos_sim(x_num_i1, x_num_i2, temp_fac)
                den_i1_i2_ss = self.cos_sim(x_num_i1, x_den, temp_fac)
                num_i1_i2_loss = -torch.log(
                    torch.exp(num_i1_i2_ss) / (torch.exp(num_i1_i2_ss) + torch.sum(tf.exp(den_i1_i2_ss))))
                net_global_loss = net_global_loss + num_i1_i2_loss
                # for positive pair (x_2,x_1);
                # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
                den_i2_i1_ss = self.cos_sim(x_num_i2, x_den, temp_fac)
                num_i2_i1_loss = -torch.log(
                    torch.exp(num_i1_i2_ss) / (torch.exp(num_i1_i2_ss) + torch.sum(tf.exp(den_i2_i1_ss))))
                net_global_loss = net_global_loss + num_i2_i1_loss

                # for positive pair (x_1,x_3);
                # numerator of loss term (num_i1_i3_ss) & denominator of loss term (den_i1_i3_ss) & loss (num_i1_i3_loss)
                num_i1_i3_ss = self.cos_sim(x_num_i1, x_num_i3, temp_fac)
                den_i1_i3_ss = self.cos_sim(x_num_i1, x_den, temp_fac)
                num_i1_i3_loss = -torch.log(
                    torch.exp(num_i1_i3_ss) / (torch.exp(num_i1_i3_ss) + torch.sum(tf.exp(den_i1_i3_ss))))
                net_global_loss = net_global_loss + num_i1_i3_loss
                # for positive pair (x_3,x_1);
                # numerator same & denominator of loss term (den_i3_i1_ss) & loss (num_i3_i1_loss)
                den_i3_i1_ss = self.cos_sim(x_num_i3, x_den, temp_fac)
                num_i3_i1_loss = -torch.log(
                    torch.exp(num_i1_i3_ss) / (torch.exp(num_i1_i3_ss) + torch.sum(tf.exp(den_i3_i1_ss))))
                net_global_loss = net_global_loss + num_i3_i1_loss

                # for positive pair (x_2,x_3);
                # numerator of loss term (num_i2_i3_ss) & denominator of loss term (den_i2_i3_ss) & loss (num_i2_i3_loss)
                num_i2_i3_ss = self.cos_sim(x_num_i2, x_num_i3, temp_fac)
                den_i2_i3_ss = self.cos_sim(x_num_i2, x_den, temp_fac)
                num_i2_i3_loss = -torch.log(
                    torch.exp(num_i2_i3_ss) / (torch.exp(num_i2_i3_ss) + torch.sum(tf.exp(den_i2_i3_ss))))
                net_global_loss = net_global_loss + num_i2_i3_loss
                # for positive pair (x_3,x_2):
                # numerator same & denominator of loss term (den_i3_i2_ss) & loss (num_i3_i2_loss)
                den_i3_i2_ss = self.cos_sim(x_num_i3, x_den, temp_fac)
                num_i3_i2_loss = -torch.log(
                    torch.exp(num_i2_i3_ss) / (torch.exp(num_i2_i3_ss) + torch.sum(tf.exp(den_i3_i2_ss))))
                net_global_loss = net_global_loss + num_i3_i2_loss
        if(global_loss_exp_no == 1):
            bs = 3 * self.batch_size
            reg_cost = net_global_loss / bs
        return
    #TODO finish function with return

    def decoder_network(in_channels, out_channels, kernel_size, padding, num_layers = 5):

            decoder_block1 = nn.Sequential(
                nn.Upsample = (in_channels, scale_factor = 2)
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

            decoder_block2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                #nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )

            dec_layer = nn.Sequential(
            decoder_block1, torch.cat(), decoder_block2
            )
            return decoder_network
    def seg_unet(self, learn_rate_seg=0.001,dsc_loss=2,en_1hot=0,mtask_en=1,fs_de=2):
        # Define the U-Net (Encoder & Decoder Network) to segment the input image):

        # Last layer from Encoder network (e)
        enc_c6_b, enc_layers_list = self.encoder_network(x, train_phase, no_filters,encoder_list_return=1)

        # skip-connection layers from encoder
        enc_c1_b, enc_c2_b, enc_c3_b, enc_c4_b, enc_c5_b = enc_layers_list[0], enc_layers_list[1], enc_layers_list[2], \
                                                           enc_layers_list[3], enc_layers_list[4]
        #Decoder network
        decoder
