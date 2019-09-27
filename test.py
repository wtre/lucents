

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
import numpy as np

import os

from model_rgbd import Model_rgbd, resize2d, resize2dmask
from data import getTestingDataOnly, getTranslucentData
from utils import  DepthNorm, colorize, save_error_image


def test_model(save_dir, save_img=True, evaluate=True):

    if not os.path.exists('%s/testimg' % save_dir):
        os.makedirs('%s/testimg' % save_dir)

    # load saved model
    model = Model_rgbd().cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_overtraining.pth')))
    model.eval()
    print('model loaded for evaluation.')

    # Load data
    test_loader = getTestingDataOnly(batch_size=2)
    train_loader_l, test_loader_l = getTranslucentData(batch_size=1)

    with torch.cuda.device(0):
        model.eval()

        tot_len = len(test_loader_l)    # min(len(test_loader), len(test_loader_l))
        testiter = iter(test_loader)
        testiter_l = iter(test_loader_l)

        for i in range(tot_len):
            # print("Iteration "+str(i)+". loop start:")
            try:
                sample_batched = next(testiter)
                sample_batched_l = next(testiter_l)
            except StopIteration:
                print('  (almost) end of iteration: %d.' % i)
                break
            print('/=/=/=/=/=/ iter %02d /=/=/=/=/' % i)

            # (1) Pretext task : test and save
            image_nyu = torch.autograd.Variable(sample_batched['image'].cuda())
            depth_nyu = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            mask_raw = torch.autograd.Variable(sample_batched_l['mask'].cuda())

            depth_nyu_n = DepthNorm(depth_nyu)

            # # Apply random mask to it
            ordered_index = list(range(depth_nyu.shape[0])) # NOTE: NYU test batch size shouldn't be bigger than lucent's.
            mask_new = mask_raw[ordered_index, :, :, :]
            depth_nyu_masked = resize2d(depth_nyu_n, (480, 640)) * mask_new

            if i <= 1:
                print('====/ %02d /====' % i)
                print(image_nyu.shape)
                print(" " + str(torch.max(image_nyu)) + " " + str(torch.min(image_nyu)))
                print(depth_nyu.shape)
                print(" " + str(torch.max(depth_nyu)) + " " + str(torch.min(depth_nyu)))
                print(mask_new.shape)
                print(" " + str(torch.max(mask_new)) + " " + str(torch.min(mask_new)))

            # Predict
            depth_out_t1 = DepthNorm( model(image_nyu, depth_nyu_masked) )

            dn_resized = resize2d(depth_nyu, (240, 320))

            if save_img:
                # Save image
                vutils.save_image(depth_out_t1, '%s/testimg/1out_%02d.png' % (save_dir, i), normalize=True, range=(0, 1000))
                if not os.path.exists('%s/testimg/1in_000000_%02d.png' % (save_dir, i)):
                    vutils.save_image(depth_nyu_masked, '%s/testimg/1in_%02d.png' % (save_dir, i), normalize=True, range=(0, 1000))
                save_error_image(depth_out_t1 - dn_resized, '%s/testimg/1diff_%02d.png' % (save_dir, i), normalize=True, range=(-300, 300))

            del image_nyu, depth_nyu, depth_out_t1, dn_resized

            # (2) Main task : test and save
            image = torch.autograd.Variable(sample_batched_l['image'].cuda())
            depth_in = torch.autograd.Variable(sample_batched_l['depth_raw'].cuda())
            htped_in = DepthNorm(depth_in)

            depth_gt = torch.autograd.Variable(sample_batched_l['depth_truth'].cuda(non_blocking=True))

            depth_out_t2 = DepthNorm( model(image, htped_in) )

            mask_small = resize2dmask(mask_raw, (240, 320))
            # if i > 0 and i < 3:
            #     print('====//====')
            #     print(true_y.shape)
            #     print(pred_y.shape)
            #     print(mask_y.shape)
            print(" " + str(torch.max(depth_out_t2)) + " " + str(torch.min(depth_out_t2)))
            print(" " + str(torch.max(depth_gt)) + " " + str(torch.min(depth_gt)))
            if i == 0:
                (s0, _, s2, s3) = depth_out_t2.size()
                # https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array
                true_y = np.empty((s0, 0, s2, s3), float)
                pred_y = np.empty((s0, 0, s2, s3), float)
                mask_y = np.empty((s0, 0, s2, s3), float)
            if evaluate:
                true_y = np.append(true_y, depth_gt.cpu().numpy(), axis=1)
                pred_y = np.append(pred_y, depth_out_t2.detach().cpu().numpy(), axis=1)
                mask_y = np.append(mask_y, mask_small.cpu().numpy(), axis=1)

            # dl = depth_in.cpu().numpy()
            # hl = htped_in.cpu().numpy()
            # dr = resize2d(depth_in, (240, 320)).cpu().numpy()
            # hr = resize2d(htped_in, (240, 320)).cpu().numpy()
            # do = depth_out_t2.cpu().detach().numpy()
            # gr = depth_gt.cpu().numpy()
            #
            # print("  Depth input (original size):" + str(np.min(dl)) + "~" + str(np.max(dl)) + " (" + str(np.mean(dl)) + ")")
            # print("  Depth Normed (original size):" + str(np.min(hl)) + "~" + str(np.max(hl)) + " (" + str(np.mean(hl)) + ")")
            #
            # print("  Depth input (resized):" + str(np.min(dr)) + "~" + str(np.max(dr)) + " (" + str(np.mean(dr)) + ")")
            # print("  Depth Normed (resized):" + str(np.min(hr)) + "~" + str(np.max(hr)) + " (" + str(np.mean(hr)) + ")")
            #
            # print("  Output converted to depth:" + str(np.min(do)) + "~" + str(np.max(do)) + " (" + str(np.mean(do)) + ")")
            # print("  GT depth (original size):" + str(np.min(gr)) + "~" + str(np.max(gr)) + " (" + str(np.mean(gr)) + ")")

            if save_img:
                if not os.path.exists('%s/testimg/2truth_000000_%02d.png' % (save_dir, i)):
                    vutils.save_image(depth_in, '%s/testimg/2inDS_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                    vutils.save_image(DepthNorm(htped_in), '%s/testimg/2inFF_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                    vutils.save_image(depth_gt, '%s/testimg/2truth_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                vutils.save_image(depth_out_t2, '%s/testimg/2out_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                save_error_image(resize2d(depth_out_t2, (480, 640)) - depth_in, '%s/testimg/2corr_%02d.png'
                                 % (save_dir, i), normalize=True, range=(-50, 50), mask=mask_raw)
                save_error_image(depth_out_t2 - depth_gt, '%s/testimg/2diff_%02d.png' % (save_dir, i), normalize=True, range=(-50, 50), mask=mask_small)
                vutils.save_image(mask_small, '%s/testimg/2_mask_%02d.png' % (save_dir, i), normalize=True, range=(-0.5, 1.5))
            del image, htped_in, depth_in, depth_gt, depth_out_t2, mask_raw, mask_small

    if evaluate:
        # true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        # pred_y = pred_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

        print(len(true_y))
        for j in range(len(true_y)):
            errors = compute_errors(true_y[j], pred_y[j], mask_y[j])
        print(errors)


def compute_errors(gt_, pred_, mask):
    # TODO: implement mask so there's no divide by 0
    mask_idx = np.nonzero(mask)
    gt = gt_[mask_idx]
    pred = pred_[mask_idx]

    thresh = np.maximum((gt / pred), (pred / gt))
    tavg = thresh.mean()
    tiavg = (1/thresh).mean()
    tmed = np.median(thresh)

    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    return a1, a2, a3, abs_rel, rmse, log_10, tavg, tiavg, tmed


if __name__ == '__main__':
    test_model('models/190925_mod12')