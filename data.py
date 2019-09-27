import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}

def loadZipToMem(zip_file, csv_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data[csv_file]).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train

class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        sample = {'image': image, 'depth': depth}
        if self.transform:
            # print('original dataset-transform exists.')
            sample = self.transform(sample)
        # else:
        #     print('original dataset- no transform.')
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 10#00   # still need to examine the exact value..
        else:            
            depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

def getTrainingTestingData(batch_size):
    data, nyu2_train = loadZipToMem('nyu_data.zip', 'data/nyu2_train.csv')

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True, drop_last=True), \
           DataLoader(transformed_testing, batch_size, shuffle=False, drop_last=True)

# =============================================================================
# custom functions below
# =============================================================================

def getTestingDataOnly(batch_size):
    data, nyu2_test = loadZipToMem('nyu_data.zip', 'data/nyu2_test.csv')
    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform(is_test=True))

    return DataLoader(transformed_testing, batch_size, shuffle=False, drop_last=True)


class RandomHorizontalFlip_l(object):
    def __call__(self, sample):
        image, depth_raw, mask, depth_truth = sample['image'], sample['depth_raw'], sample['mask'], sample['depth_truth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        # if not _is_pil_image(depth):
        #     raise TypeError(
        #         'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth_raw = depth_raw.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            depth_truth = depth_truth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth_raw': depth_raw, 'mask': mask, 'depth_truth': depth_truth}



class RandomChannelSwap_l(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth_raw, mask, depth_truth = sample['image'], sample['depth_raw'], sample['mask'], sample['depth_truth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        # if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth_raw': depth_raw, 'mask': mask, 'depth_truth': depth_truth}


class ToTensor_l(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth_raw, mask, depth_truth = sample['image'], sample['depth_raw'], sample['mask'], sample['depth_truth']

        image = image.crop((0, 14, 512, 398)).resize((640, 480))
        mask = mask.crop((0, 14, 512, 398)).resize((640, 480)) # set as 'PIL.IMage.NEARST' by default.

        image = self.to_tensor(image)
        mask, _, _ = mask.split()   # this needs to be dealt with data gathering script though.
        mask = self.to_tensor(mask)

        depth_raw = depth_raw.crop((0, 14, 512, 398)).resize((640, 480))
        depth_truth = depth_truth.crop((0, 14, 512, 398)).resize((320, 240))

        # if self.is_test:
        #     depth_raw = self.to_tensor(depth_raw).float() / 1000
        #     depth_truth = self.to_tensor(depth_truth).float() / 1000
        # else:
        #     depth_raw = self.to_tensor(depth_raw).float() * 1000
        #     depth_truth = self.to_tensor(depth_truth).float() * 1000

        # Note that our image is uint16, and NYU_v2 >train< is uint8. Hence this.
        depth_raw = self.to_tensor(depth_raw).float() / 10
        depth_truth = self.to_tensor(depth_truth).float() / 10

        # put in expected range
        depth_raw = torch.clamp(depth_raw, 0, 5000)
        depth_truth = torch.clamp(depth_truth, 0, 5000)     # Minimum value is 0, instead of 10.
        mask = torch.clamp(mask, 0, 1)

        return {'image': image, 'depth_raw': depth_raw, 'mask': mask, 'depth_truth': depth_truth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def getNoLucentTransform(is_test=False):
    return transforms.Compose([
        ToTensor_l(is_test=is_test)
    ])

def getLucentTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip_l(),
        RandomChannelSwap_l(0.5),
        ToTensor_l()
    ])


class LucentDatasetMemory(Dataset):
    def __init__(self, data, lucent_train, transform=None):
        self.data, self.lucent_dataset = data, lucent_train
        self.transform = transform
        # https://discuss.pytorch.org/t/typeerror-batch-must-contain-tensors-numbers-dicts-or-lists-found-object/14665/3
        # do ToTensor()(sample) !! https://pytorch.org/docs/master/torchvision/transforms.html

    def __getitem__(self, idx):
        sample = self.lucent_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth_raw = Image.open( BytesIO(self.data[sample[1]]) )
        mask = Image.open( BytesIO(self.data[sample[2]]) )
        depth_truth = Image.open( BytesIO(self.data[sample[3]]) )
        sample = {'image': image, 'depth_raw': depth_raw,
                  'mask': mask, 'depth_truth': depth_truth}
        # sample = {'image': self.transform(image), 'depth_raw': self.transform(depth_raw),
        #           'mask': self.transform(mask), 'depth_truth': self.transform(depth_truth)} # transform added
        if self.transform:
            # print('transform exists.')
            sample = self.transform(sample)
        else:
            print('no transform.')
        return sample

    def __len__(self):
        return len(self.lucent_dataset)

def getTranslucentData(batch_size):
    data, lucent_train = loadZipToMem('lucents_v1_moretest.zip', 'data/train.csv')
    data_, lucent_test = loadZipToMem('lucents_v1_moretest.zip', 'data/test.csv')

    transformed_training = LucentDatasetMemory(data, lucent_train, transform=getLucentTrainTransform())
    transformed_testing = LucentDatasetMemory(data_, lucent_test, transform=getNoLucentTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True, drop_last=True), \
           DataLoader(transformed_testing, batch_size*2, shuffle=False, drop_last=True)     # Note that test batch is manually enlarged!
