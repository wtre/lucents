

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
import numpy as np

import os

from model_rgbd import Model_rgbd, resize2d, resize2dmask
from data import getTestingDataOnly, getTranslucentData
from utils import  DepthNorm, thresh_mask, save_error_image


def test_model(save_dir, save_img=True, evaluate=True):

    if not os.path.exists('%s/testimg' % save_dir):
        os.makedirs('%s/testimg' % save_dir)

    # load saved model
    model = Model_rgbd().cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, 'models_asitwas/model_overtraining.pth')))
    model.eval()
    print('model loaded for evaluation.')

    # Load data
    test_loader = getTestingDataOnly(batch_size=2)
    train_loader_l, test_loader_l = getTranslucentData(batch_size=1)
    with torch.no_grad():
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

                # if i <= 1:
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
                obj_mask = thresh_mask(depth_gt, resize2d(depth_in, (240, 320)))
                # print(" " + str(torch.max(depth_out_t2)) + " " + str(torch.min(depth_out_t2)))
                # print(" " + str(torch.max(depth_gt)) + " " + str(torch.min(depth_gt)))
                # print(" " + str(torch.max(depth_in)) + " " + str(torch.min(depth_in)))
                if i == 0:
                    (s0, s1, s2, s3) = depth_out_t2.size()
                    # https://stackoverflow.com/questions/22392497/how-to-add-a-new-row-to-an-empty-numpy-array
                    true_y = np.empty((0, s1, s2, s3), float)
                    raw_y = np.empty((0, s1, s2, s3), float)
                    pred_y = np.empty((0, s1, s2, s3), float)
                    mask_y = np.empty((0, s1, s2, s3), float)
                    objmask_y = np.empty((0, s1, s2, s3), float)
                if evaluate:
                    true_y = np.append(true_y, depth_gt.cpu().numpy(), axis=0)
                    raw_y = np.append(raw_y, resize2d(depth_in, (240, 320)).cpu().numpy(), axis=0)
                    pred_y = np.append(pred_y, depth_out_t2.detach().cpu().numpy(), axis=0)
                    mask_y = np.append(mask_y, mask_small.cpu().numpy(), axis=0)
                    objmask_y = np.append(objmask_y, obj_mask.cpu().numpy(), axis=0)

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
                        vutils.save_image(depth_in, '%s/testimg/2in_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                        vutils.save_image(resize2d(depth_in, (240, 320)), '%s/testimg/2in_s_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                        vutils.save_image(depth_gt, '%s/testimg/2truth_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                    vutils.save_image(depth_out_t2, '%s/testimg/2out_%02d.png' % (save_dir, i), normalize=True, range=(0, 500))
                    save_error_image(resize2d(depth_out_t2, (480, 640)) - depth_in, '%s/testimg/2corr_%02d.png'
                                     % (save_dir, i), normalize=True, range=(-50, 50), mask=mask_raw)
                    save_error_image(depth_out_t2 - depth_gt, '%s/testimg/2diff_%02d.png' % (save_dir, i), normalize=True, range=(-50, 50), mask=mask_small)
                    vutils.save_image(mask_small, '%s/testimg/2_mask_%02d.png' % (save_dir, i), normalize=True, range=(-0.5, 1.5))
                    vutils.save_image(obj_mask, '%s/testimg/2_objmask_%02d.png' % (save_dir, i), normalize=True, range=(-0.5, 1.5))
                del image, htped_in, depth_in, depth_gt, depth_out_t2, mask_raw, mask_small

    if evaluate:

        eo = eo_r = 0
        print('#    \ta1    \ta2    \ta3    \tabsrel\trmse  \tlog10 | \timprovements--> ')
        for j in range(len(true_y)):
            # errors = compute_errors(true_y[j], pred_y[j], mask_y[j])
            errors_object = compute_errors(true_y[j], pred_y[j], mask_y[j]*objmask_y[j])
            # errors_r = compute_errors(true_y[j], raw_y[j], mask_y[j])
            errors_object_r = compute_errors(true_y[j], raw_y[j], mask_y[j] * objmask_y[j])

            eo = eo + errors_object
            eo_r = eo_r + errors_object_r

            print('{j:2d} | \t'
                  '{e[1]:.4f}\t''{e[2]:.4f}\t''{e[3]:.4f}\t'
                  '{e[4]:.4f}\t''{e[5]:.3f}\t''{e[6]:.4f} | \t'
                  '{f1[1]:+.3f}\t''{f1[2]:+.3f}\t''{f1[3]:+.3f}\t'
                  '{f2[4]:+.3f}\t''{f2[5]:+.3f}\t''{f2[6]:+.3f}'
                  .format(j=j, e=errors_object, f1=(1-errors_object_r)/(1-errors_object)-1, f2=errors_object_r/errors_object-1))

        eo = eo / len(true_y)
        eo_r = eo_r / len(true_y)
        print('\ntotal \t'
              '{e[1]:.4f}\t''{e[2]:.4f}\t''{e[3]:.4f}\t'
              '{e[4]:.4f}\t''{e[5]:.3f}\t''{e[6]:.4f} | \t'
              '{f1[1]:+.3f}\t''{f1[2]:+.3f}\t''{f1[3]:+.3f}\t'
              '{f2[4]:+.3f}\t''{f2[5]:+.3f}\t''{f2[6]:+.3f}'
              .format(e=eo, f1=(1 - eo_r)/(1 - eo) - 1, f2=eo_r/eo - 1))
        # print(errors)
        # print(errors_object)
        # print(errors_r)
        # print(errors_object_r)


# TODO: print matrix with appropriate tabular formatting / objmask / new loss
# class Metrics:
#     def __init__(self, a0, a1, a2, a3, abs_rel, rmse, log_10):
#         self.a0 = a0
#         self.a1 = a1
#         self.a2 = a2
#         self.a3 = a3
#         self.abs_rel = abs_rel
#         self.rmse = rmse
#         self.log_10 = log_10
def compute_errors(gt_, pred_, mask):
    mask_idx = np.nonzero(mask)
    gt = gt_[mask_idx]
    pred = pred_[mask_idx]

    thresh = np.maximum((gt / pred), (pred / gt))

    a0 = (thresh < 1.0001).mean()
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    return np.array([a0, a1, a2, a3, abs_rel, rmse, log_10]) # , tavg, tiavg, tmed


if __name__ == '__main__':
    test_model('models/191125_mod16')