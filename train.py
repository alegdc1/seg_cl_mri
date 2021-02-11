import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from encoder_pretrain import EncoderPretrainNet
import pdb
from torchvision import transforms
from collections import OrderedDict

import torch.nn as nn

from gloss_dminus import GlobalLossDminus
from gloss_d import GlobalLossD

class TestDataset(Dataset):
    def __init__(self):
        self.data = np.random.randn(2000, 1, 160, 160)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# TODO config object
config = OrderedDict(
    seed=1,
    batch_size=20,
    max_epochs=20,
    n_parts=4,
    n_channels=1,
    n_classes=4,
    temp_fac=1,
    no_filters=np.array([16, 32, 64, 128, 128, 128]),
    loss="global_d",
)

dataset_train = TestDataset()
train_loader = DataLoader(dataset_train,
                          num_workers=4,
                          batch_size=config['batch_size']
                          )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

model = EncoderPretrainNet(config['n_channels'], config['n_classes'], config['no_filters'])

#if torch.cuda.device_count() > 1:
    #model = nn.DataParallel(model)

# choose specified loss function
criterion = {
    "global_dminus": lambda: GlobalLossDminus(config),
    "global_d": lambda: GlobalLossD(config),
}[config['loss']]()

model.to(device)

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

trans = transforms.Compose([
    transforms.RandomCrop(100),
    transforms.Resize(160),
    # TODO adjust brightness and contrast
    #lambda src: transforms.functional.adjust_brightness(src, 0.7),
    #lambda src: transforms.functional.adjust_contrast(src, 0.7),
])

for epoch in range(config['max_epochs']):
    print("Epoch {:03d}".format(epoch))

    model.train()
    for batch_x in tqdm(train_loader):
        # TODO: ranodm transforms

        batch_x = batch_x.float().to(device)

        # augment batches by random two separate random transforms
        # TODO: both must be different datasets
        if config['loss'] == "global_dminus":
            cat_batch = torch.cat([batch_x, trans(batch_x), trans(batch_x)])
        elif config['loss'] == "global_d":
            cat_batch = torch.cat([batch_x, trans(batch_x), trans(batch_x), batch_x, trans(batch_x), trans(batch_x)])

        pred = model(cat_batch)
        loss = criterion(pred)

        loss.backward()

        print("Current loss: %f" % loss.item())

    # TODO: Validation loss
    model.eval()
    with torch.no_grad():
        print("Doing validation...")

    # TODO: Test Loss
    model.eval()
    with torch.no_grad():
        print("Doing test...")