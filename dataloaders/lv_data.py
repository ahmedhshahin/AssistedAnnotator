import torch, cv2
import errno
import hashlib
import os
import sys
import tarfile
import numpy as np
from dataloaders.implementation import *
import torch.utils.data as data
from PIL import Image
from six.moves import urllib
import json
from dataloaders.mypath import Path
from skimage.exposure import adjust_gamma 
from scipy import misc

#### Ahmed ###
import pydicom, imageio
import config

cfg = config.hyperparams

class LV_Segmentation(data.Dataset):
    def __init__(self,
                 root=Path.db_root_dir('lv'),
                 split='val',
                 transform=None):

        self.root = root
        # _mask_dir = os.path.join(_liver_root, '3Dircadb1.1/MASKS_DICOM/liver')
        # _image_dir = os.path.join(_liver_root, '3Dircadb1.1/PATIENT_DICOM')
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        
        # TODO: create text files with train and val image ids
        ## Update: In file generate_text_files.py
        # _splits_dir = os.path.join(_liver_root, 'ImageSets')

        self.images = []
        self.masks = []

        for splt in self.split:
            with open(os.path.join(os.path.join(self.root, 'CAP_challenge_training_set/' + splt + '_cap.txt')), "r") as f:
                img_lines = f.read().splitlines()
            
            for i in range(len(img_lines[:cfg['dataset_index']])):
                _image = os.path.join(self.root, img_lines[i])
                _mask  = _image[:-4] + '.png'
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)
                # print(self.images[:10])
                # print(self.masks[:10])

        assert (len(self.images) == len(self.masks))
        print('Number of images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        temp = pydicom.read_file(self.images[index]).pixel_array.astype(np.float32)
        a, b = temp.shape[:2]
        _img = np.zeros((a,b,3))#, dtype = np.float32)
        for i in range(3):
            _img[..., i] = temp
        _target = imageio.imread(self.masks[index])[..., 0].astype(np.float32)
        if _target.max() == 255: _target /= 255
        
        sample = {'image': _img, 'gt': _target}
        # misc.imshow(sample['gt'])
        if self.transform is not None:
            sample = self.transform(sample)
        # misc.imshow(sample['gt'].numpy()[0])
        # thresholding the gt labels
        sample['img_path'] = self.images[index]
        sample['gt_path']  = self.masks[index]
        sample['with_hm'][:3] = sample['with_hm'][:3] / sample['with_hm'][:3].max()
        sample['with_hm'][:3] = sample['with_hm'][:3] * 255
        if len(np.unique(sample['crop_gt'].numpy())) != 2:
            sample['crop_gt'] = (sample['crop_gt'] > 0.5).type(torch.FloatTensor)
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
        tr.ScaleNRotate(rots=(-30, 30), scales=(.9, 1.1), semseg=True),
        tr.CropFromMask(crop_elems=('image', 'gt'), relax = 20, zero_pad=True),
        tr.FixedResize(resolutions={'crop_image': (256, 256), 'crop_gt': (256, 256)}),
        # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
        # tr.ToImage(norm_elem='extreme_points'),
        # tr.SelectRange(elem = 'crop_image', _min = -25, _max = 230),
        # tr.Normalize(elems = ['crop_image']),
        # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
        tr.AddHeatMap(elem = 'crop_image', hm_type = 'l1l2', tau = 7),
        tr.ToTensor()])

    dataset = LV_Segmentation(split=['train'], transform=composed_transforms_tr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, sample in enumerate(dataloader):
        img = sample['with_hm'].numpy()[0, :3]
        hm  = sample['with_hm'].numpy()[0, 3]
        img = np.transpose(img, (1,2,0))
        gt  = sample['crop_gt'].numpy()[0,0]
        plt.subplot(131)
        plt.imshow(img.astype('int'))
        plt.imshow(colorMaskWithAlpha(gt, 0.2))
        plt.subplot(132)
        plt.imshow(hm.astype('int'))
        plt.subplot(133)
        plt.imshow(gt)
        plt.suptitle("{0} \n {1}".format(sample['img_path'][0], sample['gt_path'][0]))
        print(img.max(), img.min())
        print(hm.max(), hm.min())
        print(np.unique(gt))
        plt.show()