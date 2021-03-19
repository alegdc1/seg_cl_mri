import torch
import torch.nn as nn
import numpy as np

import pdb


class GlobalLossD(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.batch_size = config['batch_size']
        self.n_parts = config['n_parts']
        self.temp_fac = config['temp_fac']

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    def compute_symmetrical_loss(self, xi, xj, x_den):
        loss_tmp = 0

        num_i1_i2_ss = self.cos_sim(xi, xj) / self.temp_fac
        den_i1_i2_ss = self.cos_sim(xi, x_den) / self.temp_fac
        num_i1_i2_loss = -torch.log(
            torch.exp(num_i1_i2_ss) / (torch.exp(num_i1_i2_ss) + torch.sum(torch.exp(den_i1_i2_ss))))
        loss_tmp = loss_tmp + num_i1_i2_loss
        # for positive pair (x_2,x_1);
        # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
        den_i2_i1_ss = self.cos_sim(xj, x_den) / self.temp_fac
        num_i2_i1_loss = -torch.log(
            torch.exp(num_i1_i2_ss) / (torch.exp(num_i1_i2_ss) + torch.sum(torch.exp(den_i2_i1_ss))))
        loss_tmp = loss_tmp + num_i2_i1_loss

        return loss_tmp

    def forward(self, reg_pred):

        net_global_loss = torch.zeros(1, device=reg_pred.device)
        ######################
        # G^{D} - Proposed variant
        # We split each volume into self.n_parts and select 1 image from each n_part of the volume
        # the Negative image selection is done as in G^{D-} (global_loss_exp_no=1)
        # Additionally, we match images across volumes belonging to identical partition numbers of the volumes along with matching the positive image with its augmented version.
        # Example: if positive image (x_i1) is from partition 1 of volume 1, then the paired positive image (x_j1) to match is taken from partition 1 of any other volume (excluding volume 1).
        ######################
        if (self.n_parts == 4):
            bs = 4 * self.batch_size
            if (self.batch_size != 12):
                factor = 10 * self.n_parts
            else:
                factor = self.n_parts
        elif (self.n_parts == 3):
            bs = 4 * self.batch_size + 5
            factor = self.n_parts + 2
        elif (self.n_parts == 6):
            bs = 5 * self.batch_size + 4
            factor = 2

        # loop over each pair of positive images in the batch to calculate the Net global contrastive loss over the whole batch.
        for pos_index in range(0, bs, 1):

            # indexes of positive pair of samples (x_i1,x_a_i1, x_j1,x_a_j1) - we can make 4 pairs: (x_i1,x_a_i1), (x_i1,x_j1), (x_j1,x_a_j1), (x_a_i1,x_a_j1)
            # x_a_i1, x_a_j1 are augmented versions of x_i1 and x_j1, respectively.
            num_i1 = np.arange(pos_index, pos_index + 1, dtype=np.int32)
            if (pos_index + self.n_parts >= bs):
                j = (pos_index + self.n_parts) % bs
                num_i2 = np.arange(j, j + 1, dtype=np.int32)
            else:
                num_i2 = np.arange(pos_index + self.n_parts, pos_index + self.n_parts + 1, dtype=np.int32)
            if (pos_index + 2 * self.n_parts >= bs):
                j = (pos_index + 2 * self.n_parts) % bs
                num_i3 = np.arange(j, j + 1, dtype=np.int32)
            else:
                num_i3 = np.arange(pos_index + 2 * self.n_parts, pos_index + 2 * self.n_parts + 1, dtype=np.int32)
            if (pos_index + 3 * self.n_parts >= bs):
                j = (pos_index + 3 * self.n_parts) % bs
                num_i4 = np.arange(j, j + 1, dtype=np.int32)
            else:
                num_i4 = np.arange(pos_index + 3 * self.n_parts, pos_index + 3 * self.n_parts + 1, dtype=np.int32)

            if (pos_index + 4 * self.n_parts >= bs):
                j = (pos_index + 4 * self.n_parts) % bs
                num_i5 = np.arange(j, j + 1, dtype=np.int32)
            else:
                num_i5 = np.arange(pos_index + 4 * self.n_parts, pos_index + 4 * self.n_parts + 1, dtype=np.int32)

            if (pos_index + 5 * self.n_parts >= bs):
                j = (pos_index + 5 * self.n_parts) % bs
                num_i6 = np.arange(j, j + 1, dtype=np.int32)
            else:
                num_i6 = np.arange(pos_index + 5 * self.n_parts, pos_index + 5 * self.n_parts + 1, dtype=np.int32)

            # print('n1,n2,n3,n4',num_i1,num_i2,num_i3,num_i4,num_i5,num_i6)

            # indexes of corresponding negative samples as per positive pair of samples.
            den_index_net = np.arange(0, bs, dtype=np.int32)

            ind_l = []
            for not_neg_index in range(0, factor * self.n_parts):
                if (num_i1 + not_neg_index * self.n_parts >= bs):
                    j = (num_i1 + not_neg_index * self.n_parts) % bs
                    # print('j1',j)
                    ind_l.append(j)
                else:
                    # print('j0',num_i1+k*n_parts)
                    ind_l.append(num_i1 + not_neg_index * self.n_parts)
            # print('ind_l',ind_l)
            den_indexes = np.delete(den_index_net, ind_l)
            # print('d1',den_i1,len(den_i1))

            # gather required positive samples x_1,x_2,x_3,x_4 for the numerator term
            x_num_i1 = reg_pred[num_i1]
            x_num_i2 = reg_pred[num_i2]
            x_num_i3 = reg_pred[num_i3]
            x_num_i4 = reg_pred[num_i4]
            x_num_i5 = reg_pred[num_i5]
            x_num_i6 = reg_pred[num_i6]

            # gather required negative samples x_1,x_2,x_3 for the denominator term
            # x_den = torch.gather(reg_pred, den_indexes)
            x_den = reg_pred[den_indexes]

            # calculate cosine similarity score + global contrastive loss for each pair of positive images
            # if(i%8<4):
            if (pos_index % (3 * self.n_parts) < self.n_parts):
                # # for positive pair (x_i1, x_a_i1): (i1,i2) and for positive pair (x_a_i1,x_i1);
                net_global_loss += self.compute_symmetrical_loss(x_num_i1, x_num_i2, x_den)

                # # for positive pair (x_a_i1, x_a_i2): (i2,i3) and for positive pair (x_a_i2, x_a_i1);
                net_global_loss += self.compute_symmetrical_loss(x_num_i2, x_num_i3, x_den)

                # # for positive pair (x_i1, x_j1): (i1,i4) and for positive pair (x_j1, x_i1)
                net_global_loss += self.compute_symmetrical_loss(x_num_i1, x_num_i4, x_den)

                # for positive pair (x_j1, x_a_j1): (i4,i5) and for positive pair (x_a_j1, x_j1)
                net_global_loss += self.compute_symmetrical_loss(x_num_i4, x_num_i5, x_den)

                # for positive pair (x_j1, x_a_j2): (i5,i6) and for positive pair (x_a_j2, x_j1)
                net_global_loss += self.compute_symmetrical_loss(x_num_i5, x_num_i6, x_den)

                # for positive pair (x_a_i1, x_a_j2): (i2,i5) and for positive pair (x_a_j1, x_a_i1)
                net_global_loss += self.compute_symmetrical_loss(x_num_i2, x_num_i5, x_den)

                # for positive pair (x_a_i2, x_a_j2): (i3,i6) and for positive pair (x_a_j2, x_a_i2)
                net_global_loss += self.compute_symmetrical_loss(x_num_i3, x_num_i6, x_den)

        net_global_loss /= 6 * self.batch_size
        return net_global_loss.mean()
