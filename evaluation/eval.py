import os.path
# from imutils import rotate_bound
import cv2
import numpy as np
from PIL import Image

import dataloaders.helpers as helpers
import evaluation.evaluation as evaluation
from dataloaders.implementation import *
from scipy import misc
import matplotlib.pyplot as plt
cls_preds = np.load('/home/ahmed/GitHub/DEXTR-PyTorch_edit/classes_dict.npy').item()

dextr_folder = '/home/ahmed/GitHub/DEXTR-PyTorch_edit/res/five_chls_dynamic/Results'

def eval_one_result(loader, folder, one_mask_per_image=False, mask_thres=0.5, use_void_pixels=True, custom_box=False):
    def mAPr(per_cat, thresholds):
        n_cat = len(per_cat)
        all_apr = np.zeros(len(thresholds))
        for ii, th in enumerate(thresholds):
            per_cat_recall = np.zeros(n_cat)
            for jj, categ in enumerate(per_cat.keys()):
                per_cat_recall[jj] = np.sum(np.array(per_cat[categ]) > th)/len(per_cat[categ])

            all_apr[ii] = per_cat_recall.mean()

        return all_apr.mean()

    # Allocate
    eval_result = dict()
    eval_result["all_jaccards"] = np.zeros(len(loader))
    eval_result["all_percent"] = np.zeros(len(loader))
    eval_result["meta"] = []
    eval_result["per_categ_jaccard"] = dict()

    # Iterate
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        if not one_mask_per_image:
            filename = os.path.join(folder,
                                    sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png')
            filename_dex = os.path.join(dextr_folder,
                                    sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png')
        else:
            filename = os.path.join(folder,
                                    sample["meta"]["image"][0] + '.png')
            filename_dex = os.path.join(dextr_folder,
                                    sample["meta"]["image"][0] + '.png')
        cls_pred = cls_preds[sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png'][0]
        mask = np.array(Image.open(filename)).astype(np.float32) / 255.
        dextr = np.array(Image.open(filename_dex)).astype(np.float32) / 255.
        gt = np.squeeze(helpers.tens2image(sample["gt"]))
        # gt = rotate_bound(gt, 0)
        img = (np.squeeze(sample['image'].numpy())).astype(int)
        if use_void_pixels:
            void_pixels = np.squeeze(helpers.tens2image(sample["void_pixels"]))
        if mask.shape != gt.shape:
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)
            dextr = cv2.resize(dextr, gt.shape[::-1], interpolation=cv2.INTER_CUBIC)

        # Threshold
        mask = (mask > mask_thres)
        dextr = (dextr > mask_thres)
        if use_void_pixels:
            # void_pixels = rotate_bound(void_pixels, 0)
            void_pixels = (void_pixels > 0.5)
        jac = evaluation.jaccard(gt, mask, void_pixels) * 100
        dextr_jac = evaluation.jaccard(gt, dextr, void_pixels) * 100
        plt.figure(figsize=(12,8))
        plt.subplot(232)
        plt.imshow(dextr.astype(np.bool) & np.logical_not(void_pixels.astype(np.bool)))
        plt.title("five chls jaccard: {0}".format(dextr_jac))
        plt.subplot(233)
        plt.imshow(img)
        plt.imshow(colorMaskWithAlpha((dextr.astype(np.bool) & np.logical_not(void_pixels.astype(np.bool))).astype(np.float32), 0.7, 'r'))
        plt.imshow(colorMaskWithAlpha((gt.astype(np.bool) & np.logical_not(void_pixels.astype(np.bool))).astype(np.float32), 0.7, 'g'))
        plt.title("IMG + GT (green) + pred_five_chls (red)")
        plt.subplot(234)
        plt.imshow(mask.astype(np.bool) & np.logical_not(void_pixels.astype(np.bool)))
        plt.title("prediction")
        plt.subplot(235)
        plt.imshow(gt.astype(np.bool) & np.logical_not(void_pixels.astype(np.bool)))
        plt.title("GT")
        plt.subplot(236)
        plt.imshow(img)
        plt.title("IMG + GT (green) + pred (red)")
        plt.imshow(colorMaskWithAlpha((mask.astype(np.bool) & np.logical_not(void_pixels.astype(np.bool))).astype(np.float32), 0.7, 'r'))
        plt.imshow(colorMaskWithAlpha((gt.astype(np.bool) & np.logical_not(void_pixels.astype(np.bool))).astype(np.float32), 0.7, 'g'))
        # plt.imshow(colorMaskWithAlpha(mask.astype(np.float32), 0.7, 'b'))
        plt.suptitle("Multiloss Jaccard Score: {0} -- Classification Decision: {1}".format(jac, cls_pred))
        plt.savefig('/home/ahmed/GitHub/DEXTR-PyTorch_edit/res/visual_mtl_freezed/' + sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png')
        # plt.show()
        plt.close()
        plt.clf()
        # Evaluate
        if use_void_pixels:
            eval_result["all_jaccards"][i] = evaluation.jaccard(gt, mask, void_pixels)
        else:
            eval_result["all_jaccards"][i] = evaluation.jaccard(gt, mask)
        # msk_save = mask.astype(np.bool) & np.logical_not(void_pixels.astype(np.bool))
        # misc.imsave('/home/ahmed/GitHub/DEXTR-PyTorch_edit/res/rotations/Masks0/' + sample["meta"]["image"][0] + '-' + sample["meta"]["object"][0] + '.png', msk_save.astype(int))

        if custom_box:
            box = np.squeeze(helpers.tens2image(sample["box"]))
            bb = helpers.get_bbox(box)
        else:
            bb = helpers.get_bbox(gt)

        mask_crop = helpers.crop_from_bbox(mask, bb)
        if use_void_pixels:
            non_void_pixels_crop = helpers.crop_from_bbox(np.logical_not(void_pixels), bb)
        gt_crop = helpers.crop_from_bbox(gt, bb)
        if use_void_pixels:
            eval_result["all_percent"][i] = np.sum((gt_crop != mask_crop) & non_void_pixels_crop)/np.sum(non_void_pixels_crop)
        else:
            eval_result["all_percent"][i] = np.sum((gt_crop != mask_crop))/mask_crop.size
        # Store in per category
        if "category" in sample["meta"]:
            cat = sample["meta"]["category"][0]
        else:
            cat = 1
        if cat not in eval_result["per_categ_jaccard"]:
            eval_result["per_categ_jaccard"][cat] = []
        eval_result["per_categ_jaccard"][cat].append(eval_result["all_jaccards"][i])

        # Store meta
        eval_result["meta"].append(sample["meta"])

    # Compute some stats
    eval_result["mAPr0.5"] = mAPr(eval_result["per_categ_jaccard"], [0.5])
    eval_result["mAPr0.7"] = mAPr(eval_result["per_categ_jaccard"], [0.7])
    eval_result["mAPr-vol"] = mAPr(eval_result["per_categ_jaccard"], np.linspace(0.1, 0.9, 9))
    return eval_result



