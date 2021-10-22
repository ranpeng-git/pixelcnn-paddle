import pdb
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
#from paddle.autograd import 
from layers import *
from utils import *
import numpy as np

#print(paddle.__version__)
class PixelCNNLayer_up(nn.Layer):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.Sequential([gated_resnet(nr_filters, down_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=0) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.Sequential([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []
        
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Layer):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.Sequential([gated_resnet(nr_filters, down_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.Sequential([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=2) 
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=paddle.cat((u, ul_list.pop()), 1))
        
        return u, ul

class PixelCNNLayer_down(nn.Layer):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.Sequential([gated_resnet(nr_filters, down_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=1) 
                                            for _ in range(nr_resnet)])
        
        # stream from pixels above and to thes left
        self.ul_stream = nn.Sequential([gated_resnet(nr_filters, down_right_shifted_conv2d, 
                                        resnet_nonlinearity, skip_connection=2) 
                                            for _ in range(nr_resnet) ])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=paddle.concat((u, ul_list.pop()), 1))
        
        return u, ul
         

class PixelCNN(nn.Layer):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, 
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' : 
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else : 
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.Sequential([PixelCNNLayer_down(down_nr_resnet[i], nr_filters, 
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.Sequential([PixelCNNLayer_up(nr_resnet, nr_filters, 
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.Sequential([down_shifted_conv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.Sequential([down_right_shifted_conv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.upsize_u_stream  = nn.Sequential([down_shifted_deconv2d(nr_filters, nr_filters, 
                                                    stride=(2,2)) for _ in range(2)])
        
        self.upsize_ul_stream = nn.Sequential([down_right_shifted_deconv2d(nr_filters, 
                                                    nr_filters, stride=(2,2)) for _ in range(2)])
        
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3), 
                        shift_output_down=True)

        self.ul_init = nn.Sequential([down_shifted_conv2d(input_channels + 1, nr_filters, 
                                            filter_size=(1,3), shift_output_down=True), 
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters, 
                                            filter_size=(2,1), shift_output_right=True)])
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

def forward(self, x, sample=False):
        # similar as done in the tf repo :  
        if self.init_padding is None and not sample: 
            xs = [int(y) for y in x.size()]
            x1 = paddle.ones(xs[0], 1, xs[2], xs[3])
            padding = paddle.create_parameter(shape=x1.shape, dtype=str(x1.numpy().dtype),
                                              default_initializer=paddle.nn.initializer.Assign(x1))
            padding.stop_gradient = True
            self.init_padding = padding.cuda() if x.is_cuda else padding
        
        if sample : 
            xs = [int(y) for y in x.size()]
            x2 = paddle.ones(xs[0], 1, xs[2], xs[3])
            padding = paddle.create_parameter(shape= x2.shape, dtype=str(x2.numpy().dtype),
                                              default_initializer=paddle.nn.initializer.Assign(x2))
            padding.stop_gradient = True
            padding = padding.cuda() if x.is_cuda else padding
            x = paddle.concat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else paddle.concat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2: 
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()
        
        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out

if __name__ == '__main__':
    ''' testing loss with tf version '''
    np.random.seed(1)
    xx_t = (np.random.rand(15, 32, 32, 100) * 3).astype('float32')
    yy_t  = np.random.uniform(-1, 1, size=(15, 32, 32, 3)).astype('float32')
    ############################3
    x3 = paddle.to_tensor(xx_t)
    x_t = paddle.create_parameter(shape=x3.shape,dtype=str(x3.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x3))
    x_t.stop_gradient = True
    #############################
    x4 = paddle.to_tensor(yy_t)
    y_t = paddle.create_parameter(shape=x4.shape,dtype=str(x4.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x4))
    y_t.stop_gradient = True
    loss = discretized_mix_logistic_loss(y_t, x_t)
   
    ''' testing model and deconv dimensions '''
    x =paddle.cuda_env.FloatTensor(32, 3, 32, 32).uniform_(-1., 1.)
    x5= x.cpu()
    xv = paddle.create_parameter(shape=x5.shape,dtype=str(x5.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x5))
    xv.stop_gradient = True
    ds = down_shifted_deconv2d(3, 40, stride=(2,2))

    x_v = paddle.create_parameter(shape=x.shape,dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
    x_v.stop_gradient = True

    ''' testing loss compatibility '''
    model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=x.size(1))
    model = model.cuda()
    out = model(x_v)
    loss = discretized_mix_logistic_loss(x_v, out)
    print('loss : %s' % loss.data[0])

        
