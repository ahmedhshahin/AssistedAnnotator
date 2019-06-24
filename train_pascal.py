import socket
import timeit
from datetime import datetime
import scipy.misc as sm
from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import dataloaders.config 

# PyTorch includes
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample
import torch.nn as nn
# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
import dataloaders.pascal as pascal
import dataloaders.lv_data as LV
from dataloaders import custom_transforms as tr
import networks.deeplab_resnet as resnet
from layers.loss import class_balanced_cross_entropy_loss
from dataloaders.helpers import *


cfg = config.hyperparams
# Set gpu_id to -1 to run in CPU mode, otherwise set the id of the corresponding gpu
gpu_id = cfg['gpu_id']
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU: {} '.format(gpu_id))

# Setting parameters
nEpochs = cfg['nEpochs']  # Number of epochs for training
resume_epoch = cfg['resume_epoch']  # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
classifier = 'psp'  # Head classifier to use
p['trainBatch'] = cfg['train_batch']  # Training batch size
testBatch = 1  # Testing batch size
useTest = 1  # See evolution of the test set when training?
nTestInterval = 1  # Run on test set every nTestInterval epochs
snapshot = 100  # Store a model every snapshot epochs
nInputChannels = cfg['n_channels']  # Number of input channels (RGB + heatmap of extreme points)
zero_pad_crop = True  # Insert zero padding when cropping the image
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = cfg['lr'] # Learning rate
p['wd'] = cfg['wd']  # Weight decay
p['momentum'] = 0.9  # Momentum

def restore_graphs(Writer, train_loss, val_loss, accs):
    '''This function should be called when you resume training on a pretrained model. It plots the losses and accuracies of the pretrained model
    to give a full training curves in tensorboard.
    Inputs:
        - Summary writer, stored training losses array, stored vaidation losses array, stored validation accuracies array
    Output:
        - None -- arrays are plotted in tensorboard
    '''
    n = len(train_loss)
    for i in range(n):
        Writer.add_scalars('data/loss_epoch', {'validation': val_loss[i],'training segmentation': train_loss_seg[i]}, i)
        Writer.add_scalar('data/validation_accuracy', accs[i], i)
    print("Graphs Restored")

# Results and model directories (a new directory is generated for every run)
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if resume_epoch == 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
else:
    run_id = 0
save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))
modelName = 'deep_annotator'
model_fname = os.path.join(save_dir, 'models', modelName + '_best_epoch.pth')

net = resnet.resnet101(1, pretrained=True, nInputChannels=nInputChannels, classifier=classifier)
if cfg['data_parallel']:
    net = nn.DataParallel(net, device_ids = cfg['gpu_ids'])
    train_params = [{'params': resnet.get_1x_lr_params(net.module), 'lr': p['lr']},
                    {'params': resnet.get_10x_lr_params(net.module), 'lr': p['lr'] * 10}]
else:
    train_params = [{'params': resnet.get_1x_lr_params(net), 'lr': p['lr']},
                    {'params': resnet.get_10x_lr_params(net), 'lr': p['lr'] * 10}]

if cfg['optimizer'] == 'sgd':
    optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
elif cfg['optimizer'] == 'adam':
    optimizer = optim.Adam(train_params, lr=p['lr'], weight_decay=p['wd'])
else:
    print("Only adam and sgd optimizers are implemented")


if resume_epoch == 0:
    print("Initializing from pretrained Deeplab-v2 model")
else:
    print("Initializing weights from: {}".format(model_fname))
    net, optimizer, resume_epoch, train_losses, val_losses, best_val, accs = load_checkpoint(net, optimizer, fname = model_fname, device = device)

p['optimizer'] = str(optimizer)


net.to(device)

# Training the network
if resume_epoch != nEpochs:
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    if resume_epoch == 0:
        train_losses = []
        val_losses = []
        accs = []
        best_val = 0.0
    else:
        restore_graphs(writer, train_losses, val_losses, accs)


    # Preparation of the data loaders
    if cfg['relax_dynamic_or_static'] == 'static':
        composed_transforms_tr = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-30, 30), scales=(.9, 1.1), semseg=True),
            tr.CropFromMask(crop_elems=('image', 'gt'), relax = cfg['relax_crop'], zero_pad=zero_pad_crop),
            tr.FixedResize(resolutions={'crop_image': (256, 256), 'crop_gt': (256, 256)}),
            # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
            # tr.ToImage(norm_elem='extreme_points'),
            # tr.SelectRange(elem = 'crop_image', _min = -25, _max = 230),
            # tr.Normalize(elems = ['crop_image']),
            # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
            tr.AddConfidenceMap(elem = 'crop_image', hm_type = 'l1l2', tau = 7),
            tr.ToTensor()])
        composed_transforms_ts = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), relax = cfg['relax_crop'], zero_pad=zero_pad_crop),
            tr.FixedResize(resolutions={'crop_image': (256, 256), 'crop_gt': (256, 256)}, is_val = True),
            # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
            # tr.ToImage(norm_elem='extreme_points'),
            tr.AddConfidenceMap(elem = 'crop_image', hm_type= 'l1l2', tau = 7),
            # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
            tr.ToTensor()])
    else:
        composed_transforms_tr = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25)),
            tr.CropFromMaskDynamic(crop_elems=('image', 'gt'), zero_pad=zero_pad_crop),
            tr.FixedResize(resolutions={'crop_image': (256, 256), 'crop_gt': (256, 256)}),
            # tr.ExtremePoints(sigma=10, pert=5, elem='crop_gt'),
            # tr.ToImage(norm_elem='extreme_points'),
            tr.AddConfidenceMap(elem= 'crop_image', hm_type = 'l1l2', tau = 7),
            # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
            tr.ToTensor()])
        composed_transforms_ts = transforms.Compose([
            tr.CropFromMask(crop_elems=('image', 'gt'), zero_pad=zero_pad_crop),
            tr.FixedResize(resolutions={'crop_image': (256, 256), 'crop_gt': (256, 256)}, is_val = True),
            # tr.ExtremePoints(sigma=10, pert=0, elem='crop_gt'),
            # tr.ToImage(norm_elem='extreme_points'),
            tr.AddConfidenceMap(elem = 'crop_image', hm_type= 'l1l2', tau = 7),
            # tr.ConcatInputs(elems=('crop_image', 'extreme_points')),
            tr.ToTensor()])

    voc_train = LV.LV_Segmentation(split='train', transform=composed_transforms_tr)
    voc_val = LV.LV_Segmentation(split='val', transform=composed_transforms_ts)

    p['dataset_train'] = str(voc_train)
    p['transformations_train'] = [str(tran) for tran in composed_transforms_tr.transforms]
    p['dataset_test'] = str(voc_val)
    p['transformations_test'] = [str(tran) for tran in composed_transforms_ts.transforms]

    trainloader = DataLoader(voc_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=2)

    generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    # Train variables
    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    print("Training Network")
    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['with_hm'], sample_batched['crop_gt']
            # Forward-Backward of the mini-batch
            inputs.requires_grad_()
            inputs, gts = inputs.to(device), gts.to(device)

            output = net.forward(inputs)
            output = upsample(output, size=(256, 256), mode='bilinear', align_corners=True)

            # Compute the losses, side outputs and fuse
            loss = class_balanced_cross_entropy_loss(output, gts, size_average=True, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1:
                running_loss_tr = running_loss_tr / num_img_tr
                if epoch % nTestInterval == (nTestInterval - 1):
                    train_losses.append(running_loss_tr)
                # writer.add_scalar('data/total_loss_epoch', running_loss_tr_seg, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii*p['trainBatch']+inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                tr_loss_forVisualiztion = running_loss_tr
                running_loss_tr = 0

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                # writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            net.eval()
            jac_val = 0.0
            with torch.no_grad():
                for ii, sample_batched in enumerate(testloader):
                    inputs, gts, gts_full = sample_batched['with_hm'], sample_batched['crop_gt'], sample_batched['gt']

                    # Forward pass of the mini-batch
                    inputs, gts = inputs.to(device), gts.to(device)

                    output = net.forward(inputs)
                    output = upsample(output, size=(256, 256), mode='bilinear', align_corners=True)

                    # Compute the losses, side outputs and fuse
                    loss = class_balanced_cross_entropy_loss(output, gts, size_average=True)
                    running_loss_ts += loss.item()

                    output = output.to(torch.device('cpu'))
                    # Print stuff
                    # relaxes = sample_batched['crop_relax'].numpy()
                    if cfg['relax_dynamic_or_static'] == 'static':
                        relax_crop = cfg['relax_crop']
                    else:
                        relaxes = sample_batched['crop_relax'].numpy()
                    for jj in range(int(inputs.size()[0])):
                        pred = np.transpose(output.data.numpy()[jj], (1,2,0))
                        pred = 1 / (1 + np.exp(-pred))
                        pred = np.squeeze(pred)
                        gt = tens2image(gts_full[jj])
                        if cfg['relax_dynamic_or_static'] == 'dynamic': relax_crop = relaxes[jj]
                        bbox = get_bbox(gt, pad = relax_crop, zero_pad=zero_pad_crop)
                        result = (crop2fullmask(pred, bbox, gt, zero_pad= zero_pad_crop, relax= relax_crop) > 0.5).astype(np.int)
                        if gt.max() == 255: gt /= 255
                        # void_pixels = np.squeeze(tens2image(sample_batched["void_pixels"]))
                        jac_val += calc_jaccard(gt, result)#, void_pixels)
                        

                    if ii % num_img_ts == num_img_ts - 1:
                        jac_avg = jac_val / (ii*testBatch+inputs.data.shape[0])
                        accs.append(jac_avg)
                        running_loss_ts = running_loss_ts / num_img_ts
                        val_losses.append(running_loss_ts)
                        if jac_avg > best_val:
                            print("=============================> Saving checkpoint")
                            save_checkpoint(epoch, net, optimizer, jac_avg, train_losses, val_losses, accs, model_fname)
                            best_val = jac_avg
                        print('[Epoch: %d, numImages: %5d]' % (epoch, ii*testBatch+inputs.data.shape[0]))
                        writer.add_scalars('data/loss_epoch', {'validation': running_loss_ts, 'training segmentation': tr_loss_forVisualiztion}, epoch)
                        writer.add_scalar('data/validation_accuracy', jac_avg, epoch)
                        print('Loss: %f' % running_loss_ts)
                        running_loss_ts = 0
                        print('Jaccard Accuracy: %f' % jac_avg)
                        stop_time = timeit.default_timer()
                        print("Execution time: " + str(stop_time - start_time)+"\n")
    writer.close()

