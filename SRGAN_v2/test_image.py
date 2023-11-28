import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch import nn
from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_ckp', default='checkpoint/netG_srf_4.pth', type=str, help='generator model checkpoint')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
IMAGE_NAME = opt.image_name
MODEL_CKP = opt.model_ckp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Generator(UPSCALE_FACTOR).eval()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

model.load_state_dict(torch.load(MODEL_CKP)['state_dict'])

image = Image.open('images/' + IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
image.to(device)

start = time.perf_counter()
out = model(image)
elapsed = (time.perf_counter() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('images/' + 'out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
