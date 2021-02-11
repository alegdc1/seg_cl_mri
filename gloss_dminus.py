import numpy as np
import torch
import torch.nn as nn


class GlobalLossDminus(nn.Module):
    def __init__(self, config):
        super(GlobalLossDminus, self).__init__()
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
        bs = 3 * self.batch_size
        net_global_loss = torch.zeros(1, device=reg_pred.device)

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
            rem = int(num_i1) % self.n_parts
            for not_neg_index in range(rem, bs, 4):
                ind_l.append(not_neg_index)

            # print('ind_l',ind_l)
            den_indexes = np.delete(den_index_net, ind_l)
            # print('d1',den_i1,len(den_i1))

            # gather required positive samples x_1,x_2,x_3 for the numerator term
            x_num_i1 = reg_pred[num_i1]
            x_num_i2 = reg_pred[num_i2]
            x_num_i3 = reg_pred[num_i3]

            # gather required negative samples x_1,x_2,x_3 for the denominator term
            x_den = reg_pred[den_indexes]

            # calculate cosine similarity score + global contrastive loss for each pair of positive images

            # for positive pair (x_1,x_2) and for positive pair (x_2,x_1)
            net_global_loss += self.compute_symmetrical_loss(x_num_i1, x_num_i2, x_den)

            # for positive pair (x_1,x_3) and for positive pair (x_1,x_3)
            net_global_loss += self.compute_symmetrical_loss(x_num_i1, x_num_i3, x_den)

            # for positive pair (x_2,x_3) and for positive pair (x_3,x_2)
            net_global_loss += self.compute_symmetrical_loss(x_num_i2, x_num_i3, x_den)

        net_global_loss /= 2 * self.batch_size
        return net_global_loss.mean()
