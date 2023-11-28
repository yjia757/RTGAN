import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
from torch import nn
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from data_utils import DatasetFromFolder, save_ckp, load_ckp
from loss_rtresnet import GeneratorLoss
from model_new import Generator
from torch.utils.tensorboard import SummaryWriter
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

RESIZE              = str(sys.argv[1])
NUM_EPOCHS          = int(sys.argv[2])
TRAIN_BATCH_SIZE    = int(sys.argv[3])
TRAIN_NUM_WORKERS   = int(sys.argv[4])
LR                  = float(sys.argv[5])
CKP_G               = str(sys.argv[7])
MSE_OR_VGG          = str(sys.argv[8])
TRAIN_DATA_PATH     = str(sys.argv[9])
VAL_DATA_PATH       = str(sys.argv[10])


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = DatasetFromFolder(TRAIN_DATA_PATH, resize=RESIZE)
    val_set = DatasetFromFolder(VAL_DATA_PATH, resize=RESIZE)
    train_loader = DataLoader(dataset=train_set, num_workers=TRAIN_NUM_WORKERS, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    writer = SummaryWriter(log_dir='logs/current_model')

    netG = Generator()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    
    generator_criterion = GeneratorLoss()
	
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = nn.DataParallel(netG)
        generator_criterion = nn.DataParallel(generator_criterion)
    netG.to(device)
    generator_criterion.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=LR)

    # load pre-trained model if exists, otherwise start from epoch 1
    print("Reading checkpoint...")
    netG_ckp_path = CKP_G

    if os.path.exists(netG_ckp_path):
        netG, optimizerG, start_epoch_G = load_ckp(netG_ckp_path, netG, optimizerG)
        start_epoch = start_epoch_G + 1
        results = {'iter_update': start_epoch_G*(len(train_set)//TRAIN_BATCH_SIZE), 'd_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
        print('RTResNet model restore success! Continue training with ', MSE_OR_VGG, ' loss!')
    else:
        start_epoch = 1
        results = {'iter_update': 0, 'd_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
        print('Not find any model! Training from scratch with ', MSE_OR_VGG, ' loss!')

# RTResNet training 
    print("RTResNet Training Starts!")
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()

        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
            results['iter_update'] += 1
    
            real_img = Variable(target)
            real_img.to(device)

            z = Variable(data)
            z.to(device)

            fake_img = netG(z)
    
            netG.zero_grad()
            g_loss = generator_criterion(fake_img, real_img, MSE_OR_VGG)
            if torch.cuda.device_count() > 1:
                g_loss.mean().backward()
            else: 
                g_loss.backward()
            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.data[0] * batch_size

            writer.add_scalar('Training/g_loss', running_results['g_loss'] / running_results['batch_sizes'], results['iter_update'])

            train_bar.set_description(desc='[%d/%d] Loss_G: %.8f' % (
                epoch, NUM_EPOCHS, running_results['g_loss'] / running_results['batch_sizes']))
        

        netG.eval()
        out_path = 'training_results/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'psnrs': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_off, val_on in val_bar:
                batch_size = val_off.size(0)
                valing_results['batch_sizes'] += batch_size
                off = val_off
                on = val_on

                off.to(device)
                on.to(device)

                # with value in the range 0-1, shape (bs=1, rgb_channel=3, height, width)
                # in PyTorch, images are represented as [channels, height, width]
                rt = netG(off)

                ###############
                # Calculate PSNR & SSIM on Y-channel
                ###############

                # with value in the range 0-1, take bs out, shape (rgb_channel=3, height, width)
                output = rt[0].cpu().numpy()
                gt = on[0].cpu().numpy()

                # with value in the range 0-1, rearrange to shape(height, width, channel)
                output = output.transpose(1,2,0)
                gt = gt.transpose(1,2,0)

                # with value in the range 16-235
                y_output = rgb2ycbcr(output)[:, :, 0]
                y_gt = rgb2ycbcr(gt)[:, :, 0]

                # Y range is 16-235
                psnr = peak_signal_noise_ratio(y_output, y_gt, data_range = (235.0-16.0))
                valing_results['psnrs'] += psnr
                valing_results['psnr'] = valing_results['psnrs'] / valing_results['batch_sizes']

                ssim = structural_similarity(y_output, y_gt, data_range = (235.0-16.0))
                valing_results['ssims'] += ssim
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                writer.add_scalar('Validating/psnr', valing_results['psnr'], results['iter_update'])
                writer.add_scalar('Validating/ssim', valing_results['ssim'], results['iter_update'])

                val_bar.set_description(
                    desc='[converting OFF images to RT images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
                val_images.extend([off.data.cpu().squeeze(0), on.data.cpu().squeeze(0), rt.data.cpu().squeeze(0)])
            
            val_images = torch.stack(val_images[0:30])
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                writer.add_image('epoch_%d_index_%d: psnr is %.4f, ssim is %.4f' % (epoch, index, valing_results['psnr'], valing_results['ssim']), image)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    
        # save model parameters
        checkpoint_netG = {
            'epoch': epoch,
            'state_dict': netG.state_dict(),
            'optimizer': optimizerG.state_dict()
        }
        
        if epoch % 1 == 0 and epoch != 0:
            save_ckp(checkpoint_netG, 'netG', epoch, 'rtresnet', MSE_OR_VGG)

        # save loss\scores\psnr\ssim
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 1 == 0 and epoch != 0:
            f = open('statistics/train_results.txt', 'a+')
            f.write('epoch: %d, g_loss: %.4f, psnr: %.4f, ssim: %.4f' % (epoch, results['g_loss'][epoch - start_epoch], results['psnr'][epoch - start_epoch], results['ssim'][epoch - start_epoch]) + '\n')        
            f.close()      

    writer.close()
