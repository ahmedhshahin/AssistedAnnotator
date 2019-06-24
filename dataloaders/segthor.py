import torch, cv2
import errno
import hashlib
import os
import sys
import tarfile
import numpy as np
from implementation import *
import torch.utils.data as data
from PIL import Image
from six.moves import urllib
import json
from mypath import Path
from skimage.exposure import adjust_gamma 

#### Ahmed ###
import pydicom
import pandas as pd
from scipy import misc
import imageio

class SGSegmentation(data.Dataset):
    BASE_DIR = 'SegThor'
    def __init__(self,
                 root=Path.db_root_dir('dl'),
                 split='val',
                 transform=None):

        self.root = root
        _segthor_root = os.path.join(self.root, self.BASE_DIR)
        # self.csv_file = pd.read_csv(os.path.join(_dl_root, 'DL_info.csv'))
        # _mask_dir = os.path.join(_dl_root, 'mask')
        # _image_dir = os.path.join(_dl_root, 'images_png/Images_png')
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        
        self.images = []
        self.masks = []

        for splt in self.split:
            with open(os.path.join(_segthor_root, 'train/' + splt + '_imgs.txt'), "r") as f:
                imgs_lines = f.read().splitlines()
            with open(os.path.join(_segthor_root, 'train/' + splt + '_msks.txt'), "r") as f:
                mask_lines = f.read().splitlines()

            assert len(imgs_lines) == len(mask_lines), "Inconsistent masks with images array length, check"

            for ii in range(len(mask_lines)):
                # TODO: Change jpg to dcm, in case of liver no extension needed in the file path
                _image = os.path.join(_segthor_root, imgs_lines[ii])# + ".jpg")
                _mask = os.path.join(_segthor_root, mask_lines[ii])# + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))
        print('Number of images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        _img = np.zeros((512,512,3), dtype = np.float32)
        temp = imageio.imread(self.images[index]).astype(np.int32)
        temp -= 32768
        for i in range(3):
            _img[..., i] = temp.astype(np.float32)
        _target = imageio.imread(self.masks[index]).astype(np.float32)
        
        sample = {'image': _img, 'gt': _target}
        x = self.masks[index]
        if self.transform is not None:
            sample = self.transform(sample)
        # thresholding the gt labels
        sample['id'] = x[x.rindex("/")+1:]
        # if sample['crop_gt'].max() != 0:
        #     thresh = (sample['crop_gt'].max() + sample['crop_gt'].min()) / 2
        #     sample['crop_gt'] = (torch.ge(sample['crop_gt'], thresh)).type(torch.FloatTensor)
        # print(sample)
        return sample

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import helpers as helpers
    import torch
    import custom_transforms as tr
    from torchvision import transforms

    transform = transforms.Compose([tr.ToTensor()])
    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-45, 45), scales=(.8, 1.2), semseg=True),
        # tr.CreateHeatMap(hm_type = 'l1l2', tau = 7),
        tr.CropFromMask(crop_elems=('image', 'gt'), relax=30, zero_pad=True),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
        # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
        # tr.ToImage(norm_elem='extreme_points'),
        tr.SelectRange(elem = 'crop_image', _min = -200, _max = 250),
        tr.Normalize(elems = ['crop_image']),
        # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
        tr.AddConfidenceMap(elem = 'crop_image', hm_type = 'l1l2', tau = 7),
        tr.ToTensor()])

    dataset = SGSegmentation(split=['train'], transform=composed_transforms_tr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    for i, sample in enumerate(dataloader):
        # im = sample['gt'].numpy()[0]
        # im = np.transpose(im , (1,2,0))
        # misc.imshow(im[...,0])
        s = sample['with_hm'].numpy()[0]
        im = s[:3]
        pts = s[3]
        # exts = sample['extreme_points'].numpy()[0]
        gt = sample['crop_gt'].numpy()[0,0]
        im = np.transpose(im, (1,2,0))
        # print(im.max(), im.min(), gt.max(), gt.min())
        print(sample['id'])
        plt.subplot(131)
        plt.imshow(im)
        plt.subplot(132)
        plt.imshow(im)
        plt.imshow(colorMaskWithAlpha(gt, color='r', transparency=0.5))
        plt.subplot(133)
        plt.imshow(im)
        # print(sample['pts'])
        plt.imshow(colorMaskWithAlpha(pts, color='r', transparency=0.6))
        # plt.imshow(colorMaskWithAlpha(exts[0].astype(float), color='b', transparency=0.6))
        plt.show()
        print(sample['id'])
        break
        # print(i, sample['id'][0])

    # plt.show()