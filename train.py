import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter

import os
import random
import numpy as np

# from model import Model
from model_rgbd import Model_rgbd, resize2d
from loss import ssim, grad_x, grad_y, MaskedL1, MaskedL1Grad
from data import getTrainingTestingData, getTranslucentData
from utils import AverageMeter, DepthNorm, colorize

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()
    SAVE_DIR = 'models/190820_mod6'

    with torch.cuda.device(0):

        # Create model
    #    model = Model().cuda()
    #    print('Model created.')
    # =============================================================================
        # load saved model
        # model = Model().cuda()
        # model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'epoch-19.pth')))
        # model.eval()
        # print('model loaded for evaluation.')
    # =============================================================================
        # Create RGB-D model
        model = Model_rgbd().cuda()
        print('Model created.')
    # =============================================================================

        # Training parameters
        optimizer = torch.optim.Adam( model.parameters(), args.lr )
        batch_size = args.bs
        prefix = 'densenet_' + str(batch_size)

        # Load data
        train_loader, test_loader = getTrainingTestingData(batch_size=3)
        train_loader_l, test_loader_l = getTranslucentData(batch_size=1)
        # Test batch is manually enlarged! See getTranslucentData's return.

        # Logging
        writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

        # Loss
        l1_criterion = nn.L1Loss()
        l1_criterion_masked = MaskedL1()
        grad_l1_criterion_masked = MaskedL1Grad()

        # Hand-craft loss weight of main task
        interval1 = 1
        interval2 = 2
        weight_t2loss = [.0001] * interval1 + [.000316] * interval1+ \
                        [.001] * interval1 + [.00316] * interval1 + \
                        [.01] * interval1 + [.0316] * interval1 + \
                        [.1] * interval1 + [.316] * interval1 + \
                        [1.0] * interval1 + [3.16] * interval1 + \
                        [10.0] * interval2

        # Start training...
    #    for epoch in range(args.epochs):
        for epoch in range(0, 10*interval1 + interval2):
            batch_time = AverageMeter()
            losses_nyu = AverageMeter()
            losses_lucent = AverageMeter()
            losses = AverageMeter()
            N = len(train_loader)

            # Switch to train mode
            model.train()

            end = time.time()

            # decide #(iter)
            tot_len = min(len(train_loader), len(train_loader_l))
            # print(tot_len)

            trainiter = iter(train_loader)
            trainiter_l = iter(train_loader_l)

            # print(trainiter)
            # print(trainiter_l)

            # for i, sample_batched in enumerate(zip(train_loader, train_loader_l)):
            # for i, sample_batched in enumerate(train_loader_l):
            for i in range(tot_len):
                # print("Iteration "+str(i)+". loop start:")
                try:
                    sample_batched = next(trainiter)
                    sample_batched_l = next(trainiter_l)
                except StopIteration:
                    print('  (almost) end of iteration.')
                    continue
                # print('in loop.')

                optimizer.zero_grad()

                # Prepare sample and target
                image_nyu = torch.autograd.Variable(sample_batched['image'].cuda())
                depth_nyu = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

                image_raw = torch.autograd.Variable(sample_batched_l['image'].cuda())
                mask_raw = torch.autograd.Variable(sample_batched_l['mask'].cuda())
                depth_raw = torch.autograd.Variable(sample_batched_l['depth_raw'].cuda())
                depth_gt = torch.autograd.Variable(sample_batched_l['depth_truth'].cuda(non_blocking=True))

                # if i == 10:
                #     print('========')
                #     print(image_nyu.shape)
                #     print(depth_nyu.shape)
                #     print(image_raw.shape)
                #     print(" " + str(torch.max(image_raw)) + " " + str(torch.min(image_raw)))
                #     print(mask_raw.shape)
                #     print(" " + str(torch.max(mask_raw)) + " " + str(torch.min(mask_raw)))
                #     print(depth_raw.shape)
                #     print(" " + str(torch.max(depth_raw)) + " " + str(torch.min(depth_raw)))
                #     print(depth_gt.shape)
                #     print(" " + str(torch.max(depth_gt)) + " " + str(torch.min(depth_gt)))

                N1 = image_nyu.shape[0]
                N2 = image_raw.shape[0]

                ###########################
                # (1) Pretext task: depth completion

                # Normalize depth
                depth_nyu_n = DepthNorm(depth_nyu)

                # Apply random mask to it
                rand_index = [random.randint(0, N2-1) for k in range(N1)]
                mask_new = mask_raw[rand_index, :, :, :]
                depth_nyu_masked = resize2d(depth_nyu_n, (480, 640)) * mask_new

                # Predict
                output_t1 = model(image_nyu, depth_nyu_masked)
                # print("  (1): " + str(output_task1.shape))

                if i % 150 == 0 or i < 5:
                    vutils.save_image(DepthNorm(depth_nyu_masked), '%s/img/A_masked_%06d.png' % (SAVE_DIR, epoch*10000+i), normalize=True)
                    vutils.save_image(DepthNorm(output_t1), '%s/img/A_out_%06d.png' % (SAVE_DIR, epoch*10000+i), normalize=True)

                ###########################
                # (2) Main task: Undistort translucent object

                # Normalize depth
                depth_gt_n = DepthNorm(depth_gt)
                depth_raw_n = DepthNorm(depth_raw)

                # Predict
                output_t2 = model(image_raw, depth_raw_n)
                output_t2_n = DepthNorm(output_t2)
                # print("  (2): " + str(output.shape))

                if i % 150 == 0 or i < 5:
                    vutils.save_image(depth_raw, '%s/img/B_ln_%06d.png' % (SAVE_DIR, epoch*10000+i), normalize=True, range=(0, 1000))
                    vutils.save_image(depth_gt, '%s/img/B_gt_%06d.png' % (SAVE_DIR, epoch*10000+i), normalize=True, range=(0, 1000))
                    vutils.save_image(output_t2_n, '%s/img/B_out_%06d.png' % (SAVE_DIR, epoch*10000+i), normalize=True, range=(0, 1000))
                    vutils.save_image(output_t2_n-depth_gt, '%s/img/B_zdiff_%06d.png' % (SAVE_DIR, epoch*10000+i), normalize=True, range=(-500, 500))

                if i % 150 == 0 :
                    o2 = output_t2.cpu().detach().numpy()
                    o3 = output_t2_n.cpu().detach().numpy()
                    og = depth_gt.cpu().numpy()
                    nm = DepthNorm(depth_nyu_masked).cpu().numpy()
                    ng = depth_nyu.cpu().numpy()
                    print('========')
                    print("Output_t2:" + str(np.min(o2)) + "~" + str(np.max(o2)) + " (" + str(np.mean(o2)) +
                          ") // Converted to depth: " + str(np.min(o3)) + "~" + str(np.max(o3)) + " (" + str(np.mean(o3)) + ")")
                    print("GT depth : " + str(np.min(og)) + "~" + str(np.max(og)) +
                          " // NYU GT depth from 0.0~" + str(np.max(nm)) + " to " + str(np.min(ng)) + "~" + str(np.max(ng)) + " (" + str(np.mean(ng)) + ")")

                ###########################
                # (3) Update the network parameters

                mask_post = resize2d(mask_raw, (240, 320))

                # Compute the loss
                l_depth_t1 = l1_criterion(output_t1, depth_nyu_n)
                l_grad_t1 = l1_criterion(grad_x(output_t1), grad_x(depth_nyu_n)) + l1_criterion(grad_y(output_t1), grad_y(depth_nyu_n))
                l_ssim_t1 = torch.clamp((1 - ssim(output_t1, depth_nyu_n, val_range=1000.0 / 10.0)) * 0.5, 0, 1)

                l_depth_t2 = l1_criterion_masked(output_t2, depth_gt_n, mask_post)
                l_grad_t2 = grad_l1_criterion_masked(output_t2, depth_gt_n, mask_post)
                # l_depth = l1_criterion(output*mask_post, depth_ln*mask_post)
                # l_grad = l1_criterion(grad_x(output), grad_x(depth_ln), mask_post) \
                #          + l1_criterion(grad_y(output), grad_y(depth_ln), mask_post)
                l_ssim_t2 = torch.clamp((1 - ssim(output_t2, depth_gt_n, val_range=1000.0/10.0)) * 0.5, 0, 1)

                loss_nyu = (0.1 * l_depth_t1) + (1.0* l_grad_t1) + (1.0 * l_ssim_t1)
                loss_lucent = (0.1 * l_depth_t2) + (1.0 * l_grad_t2) + (0 * l_ssim_t2)
                loss = loss_nyu + (weight_t2loss[epoch] * loss_lucent)

                # Log losses
                losses_nyu.update(loss_nyu.data.item(), image_nyu.size(0))
                losses_lucent.update(loss_lucent.data.item(), image_raw.size(0))
                losses.update(loss.data.item(), image_nyu.size(0) + image_raw.size(0))

                # Update step
                loss.backward()
                optimizer.step()

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))

                # Log progress
                niter = epoch*N+i
                if i % 15 == 0:
                    # Print to console
                    print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                    'ETA {eta}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f}) ||\t'
                    'NYU {l1.val:.4f} ({l1.avg:.4f})\t'
                    'LUC {l2.val:.4f} ({l2.avg:.4f})'
                    .format(epoch, i, N, batch_time=batch_time, loss=losses, l1=losses_nyu, l2=losses_lucent, eta=eta))

                    # Log to tensorboard
                    writer.add_scalar('Train/Loss', losses.val, niter)

                if i % 15 == 0:
                    LogProgress(model, writer, test_loader_l, niter, SAVE_DIR)
                    path = os.path.join(SAVE_DIR, 'model_overtraining.pth')
                    torch.save(model.cpu().state_dict(), path) # saving model
                    model.cuda() # moving model to GPU for further training

            # Record epoch's intermediate results
            LogProgress(model, writer, test_loader_l, niter, SAVE_DIR)
            writer.add_scalar('Train/Loss.avg', losses.avg, epoch)
            # all the saves come from https://discuss.pytorch.org/t/how-to-save-a-model-from-a-previous-epoch/20252

            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'epoch-{}.pth'.format(epoch)))

    print('Program terminated.')


# TODO: Re-examine the imageloader normalization schemes, and logically deal THA problem
def LogProgress(model, writer, test_loader_l, epoch, save_dir):
    with torch.cuda.device(0):
        model.eval()
        sequential = test_loader_l
        sample_batched = next(iter(sequential))
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth_in = torch.autograd.Variable(sample_batched['depth_raw'].cuda())
        depth_in_n = DepthNorm(depth_in)

        depth = torch.autograd.Variable(sample_batched['depth_truth'].cuda(non_blocking=True))

        # print('====//====')
        # print(image.shape)
        # print(" " + str(torch.max(image)) + " " + str(torch.min(image)))
        # print(depth_in.shape)
        # print(" " + str(torch.max(depth_in)) + " " + str(torch.min(depth_in)))
        # print(depth.shape)
        # print(" " + str(torch.max(depth)) + " " + str(torch.min(depth)))

        if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
        if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
        output = DepthNorm( model(image, depth_in_n) )
        writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
        writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)

        depth_resized = resize2d(depth_in_n, (240, 320))
        vutils.save_image(depth, '%s/img/truth_%06d.png' % (save_dir, epoch), normalize=True, range=(0, 1000))
        vutils.save_image(output, '%s/img/out_%06d.png' % (save_dir, epoch), normalize=True, range=(0, 1000))
        vutils.save_image(output - depth_resized, '%s/img/diff_%06d.png' % (save_dir, epoch), normalize=True, range=(-500, 500))
        vutils.save_image(DepthNorm(depth_resized), '%s/img/in_%06d.png' % (save_dir, epoch), normalize=True, range=(0, 1000))

        del image, depth_in_n, depth_in, depth, output, depth_resized

if __name__ == '__main__':
    main()
