import random
from pathlib import Path

from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import wandb

from decoder_pretrain import DecoderPretrainNet
from encoder_pretrain import EncoderPretrainNet
from loccont_loss import LocalContrastiveLoss
from gloss_d import GlobalLossD
from gloss_dminus import GlobalLossDminus
from seg_unet import UNet

import argparse
import pdb
import json

# arguments for training script
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="string which config to use", type=str, required=True)
parser.add_argument("--wandb_mode", help="wandb mode", type=str, default="online")
args = parser.parse_args()

class TestDataset(Dataset):
    def __init__(self, config):
        self.n_slices = config["n_slices"]
        self.n_slices = config["n_slices"]

        self.filenames = []
        # TODO: config for file ending
        for path in Path(config["datadir"]).rglob("*t1.nii.gz"):
            self.filenames.append(path)

        self.n_vols = len(self.filenames)

        self.n_parts = config['n_parts']

        # starting index of first partition of any chosen volume
        self.partition_lengths = [0]

        # find the starting and last index of each partition in a volume based on input image size. shape[0] indicates total no. of slices in axial direction of the input image.
        for k in range(1, self.n_parts + 1):
            self.partition_lengths.append(k * int(self.n_slices / self.n_parts))

    def sample_minibatch_for_global_loss(self, idx):
        vol_file = self.filenames[idx]
        volume = nib.load(vol_file).get_fdata()

        # Now sample 1 image from each partition randomly. Overall, n_parts images for each chosen volume id.
        idces = []
        for k in range(0, len(self.partition_lengths) - 1):
            # sample image from each partition randomly
            i_sel = random.sample(range(self.partition_lengths[k], self.partition_lengths[k + 1]), 1)
            # print('k,i_sel',k+count, i_sel)
            idces.append(i_sel[0])

        return volume[:, :, idces].transpose(2, 0, 1)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if config['loss'] == "global_d":
            return (
            self.sample_minibatch_for_global_loss(idx), self.sample_minibatch_for_global_loss((idx + 20) % self.n_vols))
        else:
            return self.sample_minibatch_for_global_loss(idx)


# TODO: select config via argparse
with open(args.config) as f:
    config = json.load(f)
config = OrderedDict(config)
# TODO: Code from Ernesto MutableNamedTuple

dataset_train = TestDataset(config)
train_loader = DataLoader(dataset_train,
                          num_workers=4,
                          batch_size=config['batch_size'] // config['n_parts'],
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True
                          )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# choose model from config
model = {
    "encoder_pretrain": lambda: EncoderPretrainNet(config),
    "decoder_pretrain": lambda: DecoderPretrainNet(config),
    "seg_unet": lambda: UNet(config),
}[config['model']]()

print("Running model %s" % config["model"])

# not working due to structure of batches
# if torch.cuda.device_count() > 1:
# model = nn.DataParallel(model)

# choose specified optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# choose specified loss function
criterion = {
    "global_dminus": lambda: GlobalLossDminus(config),
    "global_d": lambda: GlobalLossD(config),
    "loccont_loss": lambda: LocalContrastiveLoss(config),
    "crossentropy": lambda: nn.CrossEntropyLoss(torch.tensor([0.5, 0.5, 0.5, 0.5]).to(device)),
}[config['loss']]()

# init W&B
print("Using W&B in %s mode" % args.wandb_mode)
wandb.init(project=config["model"], mode=args.wandb_mode)
wandb.config = config

model.to(device)

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

trans = transforms.Compose([
    transforms.CenterCrop(config['crop_size']),
    transforms.Resize(config['resize_size']),
    # TODO adjust brightness and contrast
    # lambda src: transforms.functional.adjust_brightness(src, 0.7),
    # lambda src: transforms.functional.adjust_contrast(src, 0.7),
])

steps = 0
for epoch in range(config['max_epochs']):
    print("Epoch {:03d}".format(epoch))

    model.train()
    # TODO: for seg unet also get batch_y data
    for batch_x in tqdm(train_loader):
        # TODO: random transforms

        # augment batches by random two separate random transforms
        if config['loss'] == "global_dminus":
            train_batch = batch_x.float().to(device)
            train_batch = train_batch.view((-1, config["n_channels"], *config['img_size']))
            train_batch = torch.cat([train_batch, trans(train_batch), trans(train_batch)])
        elif config['loss'] == "global_d":
            batch_x1, batch_x2 = batch_x

            batch_x1 = batch_x1.float().to(device)
            batch_x1 = batch_x1.view((-1, config["n_channels"], *config['img_size']))

            batch_x2 = batch_x2.float().to(device)
            batch_x2 = batch_x2.view((-1, config["n_channels"], *config['img_size']))

            train_batch = torch.cat(
                [batch_x1, trans(batch_x1), trans(batch_x1), batch_x2, trans(batch_x2), trans(batch_x2)])
        else:
            train_batch = batch_x.float().to(device)
            train_batch = train_batch.view((-1, config["n_channels"], *config["img_size"]))

        optimizer.zero_grad(set_to_none=True)
        pred = model(train_batch)

        # this is loss for pretraining
        if isinstance(model, EncoderPretrainNet) or isinstance(model, DecoderPretrainNet):
            loss = criterion(pred)
        else:
            #TODO: for testing using same class (=1), need to get correct mask
            loss = criterion(pred, torch.ones((pred.shape[0], *config["resize_size"]), device=device).long())

        wandb.log({"loss": loss.item(), 'steps': steps})
        steps += 1

        loss.backward()
        optimizer.step()
        print("Current loss: %f" % loss.item())

    # TODO: Validation loss
    model.eval()
    with torch.no_grad():
        print("Doing validation...")

    # TODO: Test Loss
    model.eval()
    with torch.no_grad():
        print("Doing test...")
