

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
import numpy as np

import os

from model_rgbd import Model_rgbd, resize2d, resize2dmask
from data import getTestingDataOnly, getTranslucentData
from utils import  DepthNorm, colorize, save_error_image


def test_model(save_dir):

    if not os.path.exists('%s/testimg' % save_dir):
        os.makedirs('%s/testimg' % save_dir)

    # load saved model
    model = Model_rgbd().cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_overtraining.pth')))
    model.eval()
    print('model loaded for evaluation.')

    # Load data
    test_loader = getTestingDataOnly(batch_size=3)
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

            # Apply random mask to it
            ordered_index = list(range(depth_nyu.shape[0])) # NOTE: NYU test batch size shouldn't be bigger than lucent's.
            mask_new = mask_raw[ordered_index, :, :, :]
            depth_nyu_masked = resize2d(depth_nyu_n, (480, 640)) * mask_new

            # if i >= 9:
            #     print('====/ %02d /====' % i)
            #     print(image_nyu.shape)
            #     print(" " + str(torch.max(image_nyu)) + " " + str(torch.min(image_nyu)))
            #     print(depth_nyu.shape)
            #     print(" " + str(torch.max(depth_nyu)) + " " + str(torch.min(depth_nyu)))
            #     print(mask_new.shape)
            #     print(" " + str(torch.max(mask_new)) + " " + str(torch.min(mask_new)))

            # Predict
            depth_out_t1 = DepthNorm( model(image_nyu, depth_nyu_masked) )

            dn_resized = resize2d(depth_nyu, (240, 320))

            # Save image

            vutils.save_image(depth_out_t1, '%s/testimg/1out_%02d.png' % (save_dir, i), normalize=True, range=(0, 1000))
            if not os.path.exists('%s/testimg/1in_000000_%02d.png' % (save_dir, i)):
                vutils.save_image(depth_nyu_masked, '%s/testimg/1in_%02d.png' % (save_dir, i), normalize=True, range=(0, 1000))
            save_error_image(depth_out_t1 - dn_resized, '%s/testimg/1diff_%02d.png' % (save_dir, i), normalize=True, range=(-300, 300))

            del image_nyu, depth_nyu, mask_raw, depth_out_t1, dn_resized

            # (2) Main task : test and save
            image = torch.autograd.Variable(sample_batched_l['image'].cuda())
            depth_in = torch.autograd.Variable(sample_batched_l['depth_raw'].cuda())
            htped_in = DepthNorm(depth_in)

            depth_gt = torch.autograd.Variable(sample_batched_l['depth_truth'].cuda(non_blocking=True))

            # print('====//====')
            # print(image.shape)
            # print(" " + str(torch.max(image)) + " " + str(torch.min(image)))
            # print(depth_in.shape)
            # print(" " + str(torch.max(depth_in)) + " " + str(torch.min(depth_in)))
            # print(depth.shape)
            # print(" " + str(torch.max(depth)) + " " + str(torch.min(depth)))

            depth_out_t2 = DepthNorm( model(image, htped_in) )

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

            if not os.path.exists('%s/testimg/2truth_000000_%02d.png' % (save_dir, i)):
                vutils.save_image(depth_in, '%s/testimg/2inDS_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                vutils.save_image(DepthNorm(htped_in), '%s/testimg/2inFF_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                vutils.save_image(depth_gt, '%s/testimg/2truth_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
            vutils.save_image(depth_out_t2, '%s/testimg/2out_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
            save_error_image(resize2d(depth_out_t2, (480, 640)) - depth_in, '%s/testimg/2corr_%02d.png' % (save_dir, i), normalize=True, range=(-50, 50))
            save_error_image(depth_out_t2 - depth_gt, '%s/testimg/2diff_%02d.png' % (save_dir, i), normalize=True, range=(-50, 50))
            del image, htped_in, depth_in, depth_gt, depth_out_t2


if __name__ == '__main__':
    test_model('models/190903_mod9')