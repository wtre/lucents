import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import copy
from model import Model

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )



def test_featuresize(half_inputsize):

    SAVE_DIR = 'models/190603_test_as_is'
    m = Model()
    m.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'epoch-19.pth')))
    original_model = m.encoder.original_model

    # double the input feature count for transition layer 1
    original_model.features.transition1.norm = nn.BatchNorm2d(384*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    original_model.features.transition1.conv = nn.Conv2d(384*2, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    # initialize E_d, as front half(~denseblock1) of pretrained ~~DenseNet~~ DenseDepth
    depth_encoder = copy.deepcopy(m.encoder.original_model.features[0:5])
    # change #(input channel) from 3 to 1
    depth_encoder.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


    if (half_inputsize):
        x = torch.randn([5, 3, 256, 320])
        y = torch.randn([5, 1, 256, 320])
    else:
        x = torch.randn([5, 3, 480, 640])
        y = torch.randn([5, 1, 480, 640])

    features = [x] #[x[:,:3,:,:]]
    features_depth = [y] #[x[:,3:,:,:]]

    # for reference: list of k
    # 0        conv0
    # 1        norm0
    # 2        relu0        >x_block0
    # 3        pool0        >x_block1
    # 4        denseblock1
    # 5        transition1    >x_block2
    # 6        denseblock2
    # 7        transition2    >x_block3
    # 8        denseblock3
    # 9        transition3
    # 10        denseblock4    >x_block4
    # 11        norm5

    # run first half of E and E_d
    ## iter over part) https://stackoverflow.com/questions/42896141/how-to-lazily-iterate-on-reverse-order-of-keys-in-ordereddict
    i1 = i2 = 0
    for k, v in list(original_model.features._modules.items())[:5]:
        features.append( v(features[-1]) )
        print(' == '+str(i1)+" == " + str(features[-1].shape))
        i1 = i1+1
    # print('AAAND..')
    for k, v in list(depth_encoder._modules.items())[:5]:
        features_depth.append( v(features_depth[-1]) )
        print(' // '+str(i2)+" // " + str(features_depth[-1].shape))
        i2 = i2 + 1

    # concatenate two outputs from each denseblock, and replace last feature with it
    concatenated_feature = torch.cat([features[5], features_depth[5]], dim=1)
    del features[-1]
    features.append( concatenated_feature )

    # run rest of the network
    for k, v in list(original_model.features._modules.items())[5:]:
        features.append( v(features[-1]) )
        print(' == ' + str(i1) + " == " + str(features[-1].shape))
        i1 = i1 + 1

    ####

    parameters = copy.deepcopy(m.decoder.parameters)  # might be dangerous

    num_features = 2208
    new_Fnum = int(num_features * 0.5)

    conv2 = nn.Conv2d(num_features, new_Fnum, kernel_size=1, stride=1, padding=1)

    up1 = UpSample(skip_input=new_Fnum // 1 + 384, output_features=new_Fnum // 2)  # // means divide and floor
    up2 = UpSample(skip_input=new_Fnum // 2 + 192, output_features=new_Fnum // 4)
    up3 = UpSample(skip_input=new_Fnum // 4 + 96 * 2, output_features=new_Fnum // 8)
    up4 = UpSample(skip_input=new_Fnum // 8 + 96 * 2, output_features=new_Fnum // 16)

    conv3 = nn.Conv2d(new_Fnum // 16, 1, kernel_size=3, stride=1, padding=1)

    x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[11]
    x_depth0, x_depth1 = features_depth[3], features_depth[4]

    x_d0 = conv2(x_block4)
    x_d1 = up1(x_d0, x_block3)
    x_d2 = up2(x_d1, x_block2)
    x_d3 = up3(x_d2, torch.cat([x_block1, x_depth1], dim=1))  # connect both features
    x_d4 = up4(x_d3, torch.cat([x_block0, x_depth0], dim=1))  # connect both features

    print('| x_d0 | ' + str(x_d0.shape))
    print('| x_d1 | ' + str(x_d1.shape))
    print('| x_d2 | ' + str(x_d2.shape))
    print('| x_d3 | ' + str(x_d3.shape))
    print('| x_d4 | ' + str(x_d4.shape))
    print('| x_d5 | ' + str(conv3(x_d4).shape))



    # return conv3(x_d4)







    return features, features_depth

if __name__ == '__main__':
    test_featuresize(half_inputsize=True)