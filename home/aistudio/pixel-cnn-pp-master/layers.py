from utils import *

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.nn.utils import weight_norm as wn
import numpy as np

class nin(nn.Layer):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        # self.lin_a = nn.utils.weight_norm(nn.Linear(dim_in, dim_out))
        #######################################
        self.lin_a = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1D(dim_out)
        self.dim_out = dim_out
    
    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        # x = x.permute(0, 2, 3, 1)
        x = paddle.transpose(x , perm = [0, 2, 3, 1] )
        shp = [int(y) for y in x.shape]
        # out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        out = self.bn(self.lin_a(paddle.reshape(x , [shp[0]*shp[1]*shp[2], shp[3]]))) #.contiguous()
        shp[-1] = self.dim_out
        # out = out.view(shp)
        out = paddle.reshape(out , shp)
        # return out.permute(0, 3, 1, 2)
        return paddle.transpose(out , perm = [0,3,1,2])


class down_shifted_conv2d(nn.Layer):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1), 
                    shift_output_down=False, norm='batch_norm'):
        super(down_shifted_conv2d, self).__init__()
        
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2D(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.Pad2D(padding=[int((filter_size[1] - 1) / 2), # pad left
                                  int((filter_size[1] - 1) / 2), # pad right
                                  filter_size[0] - 1,            # pad top
                                  0] , mode='constant')                           # pad down
        
        if norm == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2D(num_filters_out)

        if shift_output_down :
            self.down_shift = lambda x : down_shift(x, pad=nn.Pad2D(padding=[0, 0, 1, 0], mode='constant'))
    
    def forward(self, x):
        x = self.pad(x)
        #print(x,'555555555555555555555555555555555555555555')
        x = self.conv(x)
        #print(x,'666666666666666666666666666666666666666666')
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Layer):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(down_shifted_deconv2d, self).__init__()
        # self.deconv = nn.utils.weight_norm(nn.Conv2DTranspose(num_filters_in, num_filters_out, filter_size, stride,
        #                                     output_padding=1))
        self.deconv = nn.Conv2DTranspose(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1)
        self.bn = nn.BatchNorm2D(num_filters_out)
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.bn(self.deconv(x))
        xs = [int(y) for y in x.shape]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1), 
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Layer):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1), 
                    shift_output_right=False, norm='batch_norm'):
        super(down_right_shifted_conv2d, self).__init__()
        
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.Pad2D(padding=[filter_size[1] - 1, 0, filter_size[0] - 1, 0], mode='constant')
        self.conv = nn.Conv2D(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2D(num_filters_out)

        if shift_output_right :
            self.right_shift = lambda x : right_shift(x, pad=nn.Pad2D(padding=[1, 0, 0, 0], mode='constant'))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Layer):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1), 
                    shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        # self.deconv = nn.utils.weight_norm(nn.Conv2DTranspose(num_filters_in, num_filters_out, filter_size,
        #                                          stride, output_padding=1))
        #####################################################################
        self.deconv = nn.Conv2DTranspose(num_filters_in, num_filters_out, filter_size,
                                                stride, output_padding=1)
        self.bn = nn.BatchNorm2D(num_filters_out)
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.bn(self.deconv(x))
        xs = [int(y) for y in x.shape]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


'''
skip connection parameter : 0 = no skip connection 
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''
class gated_resnet(nn.Layer):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters) # cuz of concat elu
        
        if skip_connection != 0 : 
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2D(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)


    def forward(self, og_x, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None : 
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = paddle.chunk(x, 2, axis=1)
        c3 = a * F.sigmoid(b)
        return og_x + c3
