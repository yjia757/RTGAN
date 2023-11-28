import argparse
import os
from math import log10
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity 
import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

from pytorch_ssim import SSIM
from data_utils import TestDatasetFromFolder, test_display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('model_ckp', type=str, help='generator model checkpoint location')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_CKP = opt.model_ckp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR).eval()
mse_loss = nn.MSELoss()
calc_ssim = SSIM()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    mse_loss = nn.DataParallel(mse_loss)
    calc_ssim = nn.DataParallel(calc_ssim)
model.to(device)
mse_loss.to(device)
calc_ssim.to(device)

model.load_state_dict(torch.load(MODEL_CKP)['state_dict'])

test_set = TestDatasetFromFolder('/dockerx/data/test', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
with torch.no_grad():
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)

        lr_image.to(device)
        hr_image.to(device)
        
        # with value in the range 0-1, shape (bs=1, rgb_channel=3, height, width)
        # in PyTorch, images are represented as [channels, height, width]
        sr_image = model(lr_image)

        ###############
        # Calculate PSNR on Y-channel
        ###############

        # with value in the range 0-1, take bs out, shape (rgb_channel=3, height, width)
        output = sr_image[0].cpu().numpy()
        gt = hr_image[0].cpu().numpy()
  
        # with value in the range 0-1, rearrange to shape(height, width, channel)
        output = output.transpose(1,2,0)
        gt = gt.transpose(1,2,0)
        
        # with value in the range 16-235
        y_output = rgb2ycbcr(output)[:, :, 0]
        y_gt = rgb2ycbcr(gt)[:, :, 0]
        
        # Y range is 16-235
        psnr = peak_signal_noise_ratio(y_output, y_gt, data_range = (235.0-16.0))
        
        ###############
        # calcualte PSNR on RGB
        ###############

        # mse = mse_loss(sr_image, hr_image)
        # psnr = 10 * log10(1 / mse)
        
        ###############
        # Calcualte SSIM on Y-channel
        ###############
        ssim = structural_similarity(y_output, y_gt, data_range = (235.0-16.0))

        ##############
        # calcualte SSIM on RGB
        ##############
        # ssim = calc_ssim(sr_image, hr_image).item()

        test_images = torch.stack([test_display_transform()(hr_restore_img.squeeze(0)), test_display_transform()(hr_image.data.cpu().squeeze(0)), test_display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1], padding=5)

        # save psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)
        results[image_name.split('_')[0]]['ssim'].append(ssim)

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
