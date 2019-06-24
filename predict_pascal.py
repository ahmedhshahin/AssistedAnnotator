import os

import scipy.misc as sm
import imageio
import torch
from torch.nn.functional import upsample
from torch.utils.data import DataLoader
from torchvision import transforms

import dataloaders.pascal as pascal
import networks.deeplab_resnet as resnet
from dataloaders import custom_transforms as tr
from dataloaders.helpers import *
import matplotlib.pyplot as plt

gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

relax_crop = 50
zero_pad_crop = True
classifier = 'psp'
nInputChannels = 4
modelName = 'dextr_pascalbest_epoch.pth'
net = resnet.resnet101(1, pretrained=False, nInputChannels=nInputChannels, classifier=classifier, n_classification_classes= 20)
net.to(device)
net = torch.nn.DataParallel(net)
net.load_state_dict(
        torch.load(os.path.join('./run_4/models', modelName),
                   map_location=lambda storage, loc: storage)['state_dict'])
print("Model Loaded")

composed_transforms_ts = transforms.Compose([
    tr.CropFromMask(crop_elems=('image', 'gt'), relax = relax_crop, zero_pad=zero_pad_crop),
    tr.FixedResize(resolutions={'crop_image': (512, 512), 'crop_gt': (512, 512)}, is_val = True),
    # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
    # tr.ToImage(norm_elem='extreme_points'),
    tr.AddHeatMap(elem = 'crop_image', hm_type= 'l1l2', tau = 7),
    # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
    tr.ToTensor()])

voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
testloader = DataLoader(voc_val, batch_size=1, shuffle=False, num_workers=1)

net.eval()
save_dir = './run_4'
save_dir_res = os.path.join(save_dir, 'Results')
print(save_dir_res)
if not os.path.exists(save_dir_res):
    os.makedirs(save_dir_res)
correct = 0
category_names = [
                  'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
correct_per_class = np.zeros(20)
total_per_class = np.zeros(20)
classes_dict = {}
print('Testing Network')
with torch.no_grad():
    # Main Testing Loop
    for ii, sample_batched in enumerate(testloader):
        if ii % 100 == 0: print("{0} / {1}".format(ii, len(testloader)))
        inputs, gts, metas = sample_batched['with_hm'], sample_batched['gt'], sample_batched['meta']
        # Forward of the mini-batch
        inputs = inputs.to(device)
        outputs_seg, output_cls = net.forward(inputs)
        outputs_seg = upsample(outputs_seg, size=(512, 512), mode='bilinear', align_corners=True)
        outputs_seg = outputs_seg.to(torch.device('cpu'))
        output_cls = output_cls.to(torch.device('cpu'))

        for jj in range(int(inputs.size()[0])):
            pred = np.transpose(outputs_seg.data.numpy()[jj, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            gt = tens2image(gts[jj, :, :, :])
            bbox = get_bbox(gt, pad=relax_crop, zero_pad=zero_pad_crop)
            result = crop2fullmask(pred, bbox, gt, zero_pad=zero_pad_crop, relax=relax_crop)
            # Save the result, attention to the index jj
            imageio.imsave(os.path.join(save_dir_res, metas['image'][jj] + '-' + metas['object'][jj] + '.png'), (result*255).astype('uint8'))
            output_cls = output_cls[jj].data.max(0, keepdim = True)[1]
            output_cls = output_cls.numpy()
            # print(sample_batched['meta']['category'].numpy()[jj])
            total_per_class[sample_batched['meta']['category'].numpy()[jj] - 1] += 1
            if output_cls == sample_batched['meta']['category'].numpy()[jj] - 1:
                correct_per_class[sample_batched['meta']['category'].numpy()[jj] - 1] += 1
                correct += 1
            classes_dict[metas['image'][jj] + '-' + metas['object'][jj] + '.png'] = output_cls == sample_batched['meta']['category'].numpy()[jj] - 1

print(correct / 3427)
np.save('total.npy', total_per_class)
np.save('correct_per_class.npy', correct_per_class)
np.save('classes_dict.npy', classes_dict)