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
from scipy import misc

#### Ahmed ###
import pydicom


class LiverSegmentation(data.Dataset):
    BASE_DIR = 'liver_dataset'
    def __init__(self,
                 root=Path.db_root_dir('liver'),
                 split='val',
                 transform=None):

        self.root = root
        _chaos_root = os.path.join(self.root, self.BASE_DIR)
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
            with open(os.path.join(os.path.join(_chaos_root, splt + '_imgs.txt')), "r") as f:
                img_lines = f.read().splitlines()

            with open(os.path.join(os.path.join(_chaos_root, splt + '_msks.txt')), "r") as f:
                msk_lines = f.read().splitlines()

            
            for i in range(len(img_lines)):
                # _image = os.path.join(_chaos_root, line)# + ".jpg")
                _image = os.path.join(_chaos_root, img_lines[i])
                _mask = os.path.join(_chaos_root, msk_lines[i])
                assert os.path.isfile(_image)
                assert os.path.isfile(_mask)
                self.images.append(_image)
                self.masks.append(_mask)
                # print(self.images[:10])
                # print(self.masks[:10])

        assert (len(self.images) == len(self.masks))
        print('Number of images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        _img = np.zeros((512,512,3))#, dtype = np.float32)
        for i in range(3):
            temp = misc.imread(self.images[index]).astype(np.float32)
            temp -= 32768
            _img[..., i] = temp
        _target = misc.imread(self.masks[index]).astype(np.float32)
        if _target.max() == 255: _target /= 255
        
        sample = {'image': _img, 'gt': _target}
        # misc.imshow(sample['gt'])
        if self.transform is not None:
            sample = self.transform(sample)
        # misc.imshow(sample['gt'].numpy()[0])
        # thresholding the gt labels
        sample['id'] = self.images[index]
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
    print("HHI")
    transform = transforms.Compose([tr.ToTensor()])
    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-30, 30), scales=(.9, 1.1), semseg=True),
        tr.CropFromMask(crop_elems=('image', 'gt'), zero_pad=True),
        tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}),
        # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
        # tr.ToImage(norm_elem='extreme_points'),
        tr.SelectRange(elem = 'crop_image', _min = 20, _max = 250),
        tr.Normalize(elems = ['crop_image']),
        # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
        tr.AddHeatMap(elem = 'crop_image', hm_type = 'l1l2', tau = 7),
        tr.ToTensor()])

    dataset = ChaosSegmentation(split=['val'], transform=composed_transforms_tr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, sample in enumerate(dataloader):
        five = sample['with_hm'].numpy()[0]
        im = np.transpose(five[:3], (1,2,0))
        # pts = five[3]
        # fifth = adjust_gamma(five[3], 2)
        gt = sample['crop_gt'].numpy()[0]
        gt_full =sample['gt'].numpy()[0]
        # print(sample['concat'])
        # print(pts.shape)
        plt.subplot(121)
        plt.imshow(im)
        # plt.subplot(132)
        # plt.imshow(pts)
        plt.subplot(122)
        plt.imshow(im)
        # plt.imshow(fifth)
        plt.imshow(colorMaskWithAlpha(gt[0], color='r', transparency=0.7))
        # plt.title.set_text('Image with ground truth mask')
        plt.axis('on')
        plt.suptitle(sample['id'])



        print(im.max(), im.min())
        print(gt.max(), gt.min())
        # print(np.where(im == [0,0,0]))
        plt.show()