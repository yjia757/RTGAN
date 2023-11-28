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

from pytorch_ssim import SSIM
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, save_ckp, load_ckp
from loss import GeneratorLoss
from model import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Train Ray Tracing Models')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--train_batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--train_num_workers', default=4, type=int, help='number of workders for train dataloader')
parser.add_argument('--netD_lr', default=0.001, type=float, help='learning rate of netD')
parser.add_argument('--netG_lr', default=0.001, type=float, help='learning rate of netG')

if __name__ == '__main__':
    opt = parser.parse_args()

    NUM_EPOCHS = opt.num_epochs
    TRAIN_BATCH_SIZE = opt.train_batch_size
    TRAIN_NUM_WORKERS = opt.train_num_workers
    NETD_LR = opt.netD_lr
    NETG_LR = opt.netG_lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = TrainDatasetFromFolder('/dockerx/data/train')
    val_set = ValDatasetFromFolder('/dockerx/data/val')
    train_loader = DataLoader(dataset=train_set, num_workers=TRAIN_NUM_WORKERS, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    writer = SummaryWriter(log_dir='logs/current_model')

    netG = Generator()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss()
    adversarial_criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()
    calc_ssim = SSIM()

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        generator_criterion = nn.DataParallel(generator_criterion)
        adversarial_criterion = nn.DataParallel(adversarial_criterion)
        mse_loss = nn.DataParallel(mse_loss)
        calc_ssim = nn.DataParallel(calc_ssim)
    netG.to(device)
    netD.to(device)
    generator_criterion.to(device)
    adversarial_criterion.to(device)
    mse_loss.to(device)
    calc_ssim.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=NETG_LR)
    optimizerD = optim.Adam(netD.parameters(), lr=NETD_LR)

"""
    # load pre-trained model if exists, otherwise start from epoch 1
    print("Reading checkpoint...")
    netD_ckp_path = 'checkpoint/netD_srf_%d.pth' % UPSCALE_FACTOR
    netG_ckp_path = 'checkpoint/netG_srf_%d.pth' % UPSCALE_FACTOR

    if os.path.exists(netD_ckp_path) and os.path.exists(netG_ckp_path):
        netD, optimizerD, start_epoch_D = load_ckp(netD_ckp_path, netD, optimizerD)
        netG, optimizerG, start_epoch_G = load_ckp(netG_ckp_path, netG, optimizerG)
        start_epoch = start_epoch_D + 1
        results = {'batch_sizes': start_epoch_D*TRAIN_BATCH_SIZE, 'd_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
        print("Model restore success!")
    else:
        start_epoch = 1
        results = {'batch_sizes': 0, 'd_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
        print("Not find pre-trained model!")
"""

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
            results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            real_img = Variable(target)
            target_real_out = Variable(torch.rand(batch_size, 1)*0.5 + 0.7)

            real_img.to(device)
            target_real_out.to(device)

            z = Variable(data)
            target_fake_out = Variable(torch.rand(batch_size, 1) * 0.3)
            ones_const = Variable(torch.ones(batch_size, 1))

            z.to(device)
            target_fake_out.to(device)
            ones_const.to(device)

            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img)
            fake_out = netD(Variable(fake_img.data))

            d_loss = adversarial_criterion(real_out, target_real_out) + \
                     adversarial_criterion(fake_out, target_fake_out)

            if torch.cuda.device_count() > 1:
                d_loss.mean().backward(retain_graph=True)
            else:
                d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize content loss + adversarial loss
            ###########################
            netG.zero_grad()
            fake_out = netD(fake_img)
            g_loss = generator_criterion(fake_out, fake_img, real_img, ones_const)
            if torch.cuda.device_count() > 1:
                g_loss.mean().backward()
            else:
                g_loss.backward()
            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.data[0] * batch_size
            running_results['d_loss'] += d_loss.data[0] * batch_size
            running_results['d_score'] += real_out.data[0] * batch_size
            running_results['g_score'] += fake_out.data[0] * batch_size

            writer.add_scalar('Training/g_loss', running_results['g_loss'] / running_results['batch_sizes'], results['batch_sizes'])
            writer.add_scalar('Training/d_loss', running_results['d_loss'] / running_results['batch_sizes'], results['batch_sizes'])
            writer.add_scalar('Training/d_score', running_results['d_score'] / running_results['batch_sizes'], results['batch_sizes'])
            writer.add_scalar('Training/g_score', running_results['g_score'] / running_results['batch_sizes'], results['batch_sizes'])

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr

                lr.to(device)
                hr.to(device)

                sr = netG(lr)

                batch_mse = mse_loss(sr, hr)
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = calc_ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                writer.add_scalar('Validating/psnr', valing_results['psnr'], results['batch_sizes'])
                writer.add_scalar('Validating/ssim', valing_results['ssim'], results['batch_sizes'])

                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images[0:75])
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                writer.add_image('epoch_%d_index_%d' % (epoch, index), image)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        checkpoint_netD = {
            'epoch': epoch,
            'state_dict': netD.state_dict(),
            'optimizer': optimizerD.state_dict()
        }
        checkpoint_netG = {
            'epoch': epoch,
            'state_dict': netG.state_dict(),
            'optimizer': optimizerG.state_dict()
        }
        save_ckp(checkpoint_netD, 'netD', UPSCALE_FACTOR)
        save_ckp(checkpoint_netG, 'netG', UPSCALE_FACTOR)

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        f = open('statistics/' + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.txt', 'a+')
        f.write('epoch: %d, d_loss: %.3f, g_loss: %.3f, d_score: %.3f, g_score: %.3f, psnr: %.3f, ssim: %.3f' % (epoch, results['d_loss'][epoch - start_epoch], results['g_loss'][epoch - start_epoch], results['d_score'][epoch - start_epoch], results['g_score'][epoch - start_epoch], results['psnr'][epoch - start_epoch], results['ssim'][epoch - start_epoch]) + '\n')
        f.close()

    writer.close()
