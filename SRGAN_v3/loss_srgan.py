import torch
from torch import nn
from torchvision.models.vgg import vgg19
from torch.autograd import Variable
from torchvision.transforms import Normalize


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:9]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.tv_loss = TVLoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.transform = MeanShift(norm_mean = self.mean, norm_std = self.std)

    def forward(self, out_labels, out_images, target_images, ones_const, mse_or_vgg):
        # Generator adversarial loss
        adversarial_loss = self.bce_loss(out_labels, ones_const)
        # Generator content loss - vgg features loss
        out_images_vgg = self.transform(out_images)
        target_images_vgg = self.transform(target_images)
        vgg_features_loss = self.mse_loss(self.loss_network(out_images_vgg), self.loss_network(target_images_vgg))
        # Generator content loss - image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # Generator content loss - TV Loss
        tv_loss = self.tv_loss(out_images)
        
        if mse_or_vgg == 'mse':
            # Loss function for SRGAN-MSE in original paper. 
            return 0.001 * adversarial_loss + image_loss
        elif mse_or_vgg =='vgg':
            # Loss choice one: SRGAN-VGG in original paper.  
            return 0.001 * adversarial_loss + 0.006 * vgg_features_loss
        else: 
            print('Please specify using mse or vgg!')

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range = 1,
        norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(norm_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(norm_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
