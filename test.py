import torch
import pdb
from encoder import Encoder
from decoder import Decoder
from seg_unet import UNet

test_data = torch.randn((32, 16, 32, 64))
model = UNet(16, 50)

out = model(test_data)	
print("Done")

pdb.set_trace()


