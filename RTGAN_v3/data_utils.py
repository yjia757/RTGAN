from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def train_transform(new_size):
    return Compose([
        Resize(new_size),
        CenterCrop(new_size),
        ToTensor()
    ])

def save_ckp(state, network_name, epoch, rtgan_or_rtresnet, mse_or_vgg):
    f_path = 'checkpoint/%s_%d_%s_%s.pth' % (network_name, epoch, rtgan_or_rtresnet, mse_or_vgg)
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, network, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    network.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return network, optimizer, checkpoint['epoch']

class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, resize=False, test_mode=False):
        super(DatasetFromFolder, self).__init__()
        self.off_path = dataset_dir + '/OFF/'
        self.on_path = dataset_dir + '/ON/'
        self.off_filenames = [join(self.off_path, x) for x in sorted(listdir(self.off_path)) if is_image_file(x)]
        self.on_filenames = [join(self.on_path, x) for x in sorted(listdir(self.on_path)) if is_image_file(x)]
        self.train_transform = train_transform(100)
        self.resize = resize
        self.test_mode = test_mode
    def __getitem__(self, index):
        image_name = self.off_filenames[index].split('/')[-1]
        off_image = Image.open(self.off_filenames[index]).convert('RGB')
        on_image = Image.open(self.on_filenames[index]).convert('RGB')
        if self.resize:
            off_image = self.train_transform(off_image)
            on_image = self.train_transform(on_image)
        else: 
            off_image = ToTensor()(off_image)
            on_image = ToTensor()(on_image)
        if self.test_mode:
            return image_name, off_image, on_image
        else:
            return off_image, on_image
    def __len__(self):
        return len(self.off_filenames)
