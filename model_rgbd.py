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

class Decoder(nn.Module):
    def __init__(self, num_features=2208, decoder_width = 0.5):
        super(Decoder, self).__init__()

        SAVE_DIR = 'models/190603_test_as_is'
        m = Model()
        m.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'epoch-19.pth')))
        self.parameters = copy.deepcopy(m.decoder.parameters)   # might be dangerous

        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSample(skip_input=features//1 + 384, output_features=features//2) # // means divide and floor
        self.up2 = UpSample(skip_input=features//2 + 192, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 +  96*2, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 +  96*2, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features, features_depth):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[11]
        x_depth0, x_depth1 = features_depth[3], features_depth[4]

        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, torch.cat([x_block1, x_depth1], dim=1))   # connect both features
        x_d4 = self.up4(x_d3, torch.cat([x_block0, x_depth0], dim=1))   # connect both features
        return self.conv3(x_d4)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # initialize E as pretrained ~~DenseNet~~ DenseDepth
        # import torchvision.models as models
        # self.original_model = models.densenet161( pretrained=True )

        SAVE_DIR = 'models/190603_test_as_is'
        m = Model()
        m.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'epoch-19.pth')))
        self.original_model = m.encoder.original_model

        # double the input feature count for transition layer 1
        self.original_model.features.transition1.norm = nn.BatchNorm2d(384*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.original_model.features.transition1.conv = nn.Conv2d(384*2, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # initialize E_d, as front half(~denseblock1) of pretrained ~~DenseNet~~ DenseDepth
        # self.depth_encoder = models.densenet161( pretrained=True ).features[0:5]
        self.depth_encoder = copy.deepcopy(m.encoder.original_model.features[0:5])
        # change #(input channel) from 3 to 1
        self.depth_encoder.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x, y):
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
        for k, v in list(self.original_model.features._modules.items())[:5]:
            features.append( v(features[-1]) )
            # print(features[-1].shape)
        # print('AAAND..')
        for k, v in list(self.depth_encoder._modules.items())[:5]:
            features_depth.append( v(features_depth[-1]) )
            # print(features_depth[-1].shape)

        # concatenate two outputs from each denseblock, and replace last feature with it
        concatenated_feature = torch.cat([features[5], features_depth[5]], dim=1)
        del features[-1]
        features.append( concatenated_feature )

        # run rest of the network
        for k, v in list(self.original_model.features._modules.items())[5:]:
            features.append( v(features[-1]) )

        return features, features_depth


# https://discuss.pytorch.org/t/resizing-any-simple-direct-way/10316/5
def resize2d(img, size):
    with torch.no_grad():
        t = (F.adaptive_avg_pool2d(Variable(img), size)).data
    return t

def resize2dmask(mask, size):
    with torch.no_grad():
        o_in = torch.zeros(mask.size()).cuda()
        o_out = torch.zeros(size).cuda()
        t = (o_out - F.adaptive_max_pool2d(Variable(o_in - mask), size)).data
        # t = (F.adaptive_avg_pool2d(Variable(img), size)).data
    return t

# def trim2d(img, ratio):
#     with torch.no_grad():
#         (batch_size, _, _, _) = img.size()
#         t = (F.grid_sample(img, g)).data
#     return t


class Model_rgbd(nn.Module):
    def __init__(self):
        super(Model_rgbd, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, y): # x.size(1) is now 4. or maybe not, and just take 2 arguments.
        # https://stackoverflow.com/questions/691267/passing-functions-which-have-multiple-return-values-as-arguments-in-python
        # if resize_mask:
        #     y = resize2d(y, (320, 240))
        return self.decoder( *self.encoder(x, y) )

