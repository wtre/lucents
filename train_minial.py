import time
import argparse
import datetime

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import os
import random

# from model import Model
from model_rgbd import Model_rgbd, resize2d, resize2dmask
from loss import ssim, grad_x, grad_y, MaskedL1, MaskedL1Grad
from data import getTrainingTestingData, getTranslucentData
from utils import AverageMeter, DepthNorm, thresh_mask, colorize, save_error_image, blend_depth

def main():

    SAVE_DIR = 'models/191216_mod17'
    with torch.cuda.device(0):

        # Create RGB-D model
        model = Model_rgbd().cuda()
        print('Model created.')

        # Training parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, amsgrad=True)

        # Load data
        train_loader, test_loader = getTrainingTestingData(batch_size=2)
        train_loader_l, test_loader_l = getTranslucentData(batch_size=1)

        # Loss
        l1_criterion = nn.L1Loss()

        if not os.path.exists('%s/img' % SAVE_DIR):
            os.makedirs('%s/img' % SAVE_DIR)

        num_epochs = 80
        # Start training...
        for epoch in range(0, num_epochs):

            trainiter = iter(train_loader)
            trainiter_l = iter(train_loader_l)

            for i in range(60):#range(tot_len):
                # print("Iteration "+str(i)+". loop start:")
                try:
                    sample_batched = next(trainiter)
                    sample_batched_l = next(trainiter_l)
                except StopIteration:
                    # print('  (almost) end of iteration.')
                    continue

                image_nyu = torch.autograd.Variable(sample_batched['image'].cuda())
                depth_nyu = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
                mask_raw = torch.autograd.Variable(sample_batched_l['mask'].cuda())

                N1 = image_nyu.shape[0]
                N2 = mask_raw.shape[0]

                # Normalize depth
                depth_nyu_n = DepthNorm(depth_nyu)

                # Apply random mask to it
                rand_index = [random.randint(0, N2 - 1) for k in range(N1)]
                mask_new = mask_raw[rand_index, :, :, :]
                depth_nyu_masked = resize2d(depth_nyu_n, (480, 640)) * mask_new

                # Predict
                output_t1 = model(image_nyu, depth_nyu_masked)

                # Compute the loss
                loss = l1_criterion(output_t1, depth_nyu_n)

                # Update the Network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print the times
                if i % 3 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: \t{:.4f}'
                          .format(epoch, num_epochs, i + 1, 60, loss.item()))

            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'epoch-{}.pth'.format(epoch)))
    print('Program terminated.')

if __name__ == '__main__':
    main()
