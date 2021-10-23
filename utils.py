import pdb
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


from paddle.nn.utils import weight_norm as wn
import numpy as np


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.shape) - 3
    return F.elu(paddle.concat([x, -x], axis=axis))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.shape) - 1
    m = paddle.max(x, axis=axis)
    m2 = paddle.max(x, axis=axis, keepdim=True)
    return m + paddle.log(paddle.sum(paddle.exp(x - m2), axis=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.shape) - 1
    m = paddle.max(x, axis=axis, keepdim=True)
    return x - m - paddle.log(paddle.sum(paddle.exp(x - m), axis=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    x = paddle.transpose(x , perm = [0,2,3,1])
    l = paddle.transpose(l , perm = [0,2,3,1])
    xs = [int(y) for y in x.shape]
    ls = [int(y) for y in l.shape]
   
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l =  paddle.reshape(l[:, :, :, nr_mix:] , xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = paddle.clip(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    data = paddle.zeros(xs + [nr_mix])
    data = paddle.create_parameter(shape=data.shape,
                        dtype=str(data.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(data))
    data.stop_gradients = True
    x = paddle.unsqueeze(x , -1) + data

    m2 = paddle.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :] , [xs[0], xs[1], xs[2], 1, nr_mix])

    m3 = paddle.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :] , [xs[0], xs[1], xs[2], 1, nr_mix])

    means = paddle.concat((paddle.unsqueeze(means[:, :, :, 0, :] ,3), m2, m3), axis=3)
    centered_x = x - means
    inv_stdv = paddle.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    
    inner_inner_cond = paddle.cast(cdf_delta > 1e-5 , 'float32')
    inner_inner_out  = inner_inner_cond * paddle.log(paddle.clip(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = paddle.cast(x > 0.999 , 'float32')
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = paddle.cast(x < -0.999 , 'float32')
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = paddle.sum(log_probs, axis=3) + log_prob_from_logits(logit_probs)
    
    return -paddle.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    # x = x.permute(0, 2, 3, 1)
    x = paddle.transpose(x , perm = [0,2,3,1])
    # l = l.permute(0, 2, 3, 1)
    l = paddle.transpose(l, perm = [0,2,3,1])
    xs = [int(y) for y in x.shape]
    ls = [int(y) for y in l.shape]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    # l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    l = paddle.reshape(l[:, :, :, nr_mix:] , xs + [nr_mix * 2])
    means = l[:, :, :, :, :nr_mix]
    log_scales = paddle.clip(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + paddle.to_tensor(paddle.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = paddle.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    
    inner_inner_cond = paddle.cast(cdf_delta > 1e-5 ,'float32')
    inner_inner_out  = inner_inner_cond * paddle.log(paddle.clip(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = paddle.cast(x > 0.999 ,'float32')
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = paddle.cast(x < -0.999 , 'float32')
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = paddle.sum(log_probs, axis=3) + log_prob_from_logits(logit_probs)
    
    return -paddle.sum(log_sum_exp(log_probs))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = paddle.to_tensor(tensor.shape + (n,)).zero_()
    # if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.shape), tensor.unsqueeze(-1), fill_with)
    return paddle.to_tensor(one_hot)


def sample_from_discretized_mix_logistic_1d(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.shape]
    xs = ls[:-1] + [1] #[3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    # l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # for mean, scale
    l = paddle.reshape(l[:, :, :, nr_mix:] , xs + [nr_mix * 2])

    # sample mixture indicator from softmax
    temp = paddle.to_tensor(logit_probs.shape)
    # if l.is_cuda : temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - paddle.log(- paddle.log(temp))
    _, argmax = temp.max(axis=3)
   
    one_hot = to_one_hot(argmax, nr_mix)
    # sel = one_hot.view(xs[:-1] + [1, nr_mix])
    sel = paddle.reshape(one_hot , xs[:-1] + [1, nr_mix] )

    # select logistic parameters
    means = paddle.sum(l[:, :, :, :, :nr_mix] * sel, axis=4) 
    log_scales = paddle.clip(paddle.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, axis=4), min=-7.)
    u = paddle.to_tensor(means.shape)
    # if l.is_cuda : u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = paddle.to_tensor(u)
    x = means + paddle.exp(log_scales) * (paddle.log(u) - paddle.log(1. - u))
    x0 = paddle.clip(paddle.clip(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out


def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.shape]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    # l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    l =paddle.reshape(l[:, :, :, nr_mix:] ,xs + [nr_mix * 3] )
    # sample mixture indicator from softmax
    temp = paddle.to_tensr(logit_probs.shape)
    # if l.is_cuda : temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - paddle.log(- paddle.log(temp))
    _, argmax = temp.max(axis=3)
   
    one_hot = to_one_hot(argmax, nr_mix)
    # sel = one_hot.view(xs[:-1] + [1, nr_mix])
    sel = paddle.reshape(one_hot ,xs[:-1] + [1, nr_mix] )
    # select logistic parameters
    means = paddle.sum(l[:, :, :, :, :nr_mix] * sel, axis=4) 
    log_scales = paddle.clip(paddle.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, axis=4), min=-7.)
    coeffs = paddle.sum(F.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, axis=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = paddle.to_tensor(means.shape)
    # if l.is_cuda : u = u.cuda()
    # u.uniform_(1e-5, 1. - 1e-5)
    u = paddle.uniform(u , min =1e-5 , max= 1. - 1e-5 )
    u = paddle.to_tensor(u)
    x = means + paddle.exp(log_scales) * (paddle.log(u) - paddle.log(1. - u))
    x0 = paddle.clip(paddle.clip(x[:, :, :, 0], min=-1.), max=1.)
    x1 = paddle.clip(paddle.clip(
       x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = paddle.clip(paddle.clip(
       x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    # out = paddle.concat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], axis=3)
    out = paddle.concat([paddle.reshape(x0 , xs[:-1] + [1]), paddle.reshape(x1 , xs[:-1] + [1]), paddle.reshape(x2 , xs[:-1] + [1])], axis=3)
    # put back in Pytorch ordering
    # out = out.permute(0, 3, 1, 2)
    out = paddle.transpose(out , perm = [0,3,1,2])
    return out



''' utilities for shifting the image around, efficient alternative to masking convolutions '''
def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.shape]
    # when downshifting, the last row is removed 
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.Pad2D(padding=[0, 0, 1, 0],mode="constant") if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.shape]
    # when righshifting, the last column is removed 
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.Pad2D(padding=[1, 0, 0, 0],mode="constant") if pad is None else pad
    return pad(x)

def load_part_of_model(model, path):
    params = paddle.load(path)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try : 
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                print(e)
                pass
    print('added %s of params:' % (added / float(len(model.state_dict().keys()))))
