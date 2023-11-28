from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

"""
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_on_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_off_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


def save_ckp(state, network_name, upscale_factor):
    f_path = 'checkpoint/%s_srf_%d.pth' % (network_name, upscale_factor)
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, network, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    network.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return network, optimizer, checkpoint['epoch']
"""


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.on_transform = train_on_transform(crop_size)
        self.off_transform = train_off_transform(crop_size)

    def __getitem__(self, index):
        on_image = self.on_transform(Image.open(self.image_filenames[index]))
        off_image = self.off_transform(on_image)
        return off_image, on_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        on_image = Image.open(self.image_filenames[index])
        w, h = on_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        off_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        on_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        on_image = CenterCrop(crop_size)(on_image)
        off_image = off_scale(on_image)
        on_restore_img = on_scale(off_image)
        return ToTensor()(off_image), ToTensor()(on_restore_img), ToTensor()(on_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        self.off_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.on_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.off_filenames = [join(self.off_path, x) for x in listdir(self.off_path) if is_image_file(x)]
        self.on_filenames = [join(self.on_path, x) for x in listdir(self.on_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.off_filenames[index].split('/')[-1]
        off_image = Image.open(self.off_filenames[index])
        w, h = off_image.size
        on_image = Image.open(self.on_filenames[index])
#        on_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
#        on_restore_img = on_scale(off_image)
        return image_name, ToTensor()(off_image), ToTensor()(on_image)

    def __len__(self):
        return len(self.off_filenames)
