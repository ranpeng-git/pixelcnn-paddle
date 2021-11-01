import time
import os
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.vision import datasets  , transforms
from tensorboardX import SummaryWriter
from utils import * 
from model import * 
from PIL import Image
from save_img import *

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=5,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default='pixelcnn2.pdparams',
                    help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=1000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
args = parser.parse_args()

device = paddle.device.set_device('gpu:0')

# reproducibility
paddle.seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}'.format(args.lr, args.nr_resnet, args.nr_filters)
# assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
if not os.path.exists(os.path.join('runs', model_name)):
    print('{} already exists!'.format(model_name))
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':0 , 'drop_last':True} #'pin_memory':True
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

if 'mnist' in args.dataset : 
    train_loader = paddle.io.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = paddle.io.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset : 
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                 std=[0.5, 0.5, 0.5],
    #                 data_format='HWC')
    # cifar10 = datasets.Cifar10(mode='train', transform=normalize)
    train_loader = paddle.io.DataLoader(datasets.Cifar10( mode = 'train', 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = paddle.io.DataLoader(datasets.Cifar10( mode = 'test', 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, Cifar10}'.format(args.dataset))

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
# model = model.cuda()


if args.load_params:
#     load_part_of_model(model, args.load_params)
    model.set_state_dict(paddle.load(args.load_params))
    print('model parameters loaded')


scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=1, gamma=args.lr_decay)
optimizer = optim.Adam(learning_rate=scheduler, parameters=model.parameters())

def sample(model):
    # model.train(False)
    model.train()
    data = paddle.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    # data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):

            # data_v = Variable(data)
            data_v = paddle.create_parameter(shape=data.shape,
                        dtype=str(data.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(data))
            data_v.stop_gradients = True

            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
writes = 0
for epoch in range(args.max_epochs):
    # model.train(True)
    model.train()
    # torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(train_loader):
        # input = input.cuda(async=True)
        # input = Variable(input)
        input = paddle.create_parameter(shape=input.shape,
                        dtype=str(input.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(input))
        output = model(input)
        loss = loss_op(input, output)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu().numpy()[0]
#         print(loss)
#         print(train_loss)
        if (batch_idx +1) % args.print_every == 0 : 
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('train/bpd', (train_loss / deno), writes)
            print('loss : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno), 
                (time.time() - time_)))
            train_loss = 0.
            writes += 1
            time_ = time.time()
            

    # decrease learning rate
    scheduler.step()
    
    # torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        # input = input.cuda(async=True)
        # input_var = Variable(input)
        input_var = paddle.create_parameter(shape=input.shape,
                        dtype=str(input.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(input))
        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.detach().cpu().numpy()[0]
        del loss, output

    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    writer.add_scalar('test/bpd', (test_loss / deno), writes)
    print('test loss : %s' % (test_loss / deno))

    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('images'):
        os.mkdir('images')

    if (epoch + 1) % args.save_interval == 0: 
        paddle.save(model.state_dict(), 'models/{}_{}.pdparams'.format(model_name, epoch))
        print('sampling...')
#         sample_t = sample(model)
#         sample_t = rescaling_inv(sample_t)


        ####
#         save_image(sample_t,'images/{}_{}.png'.format(model_name, epoch), 
#                 nrow=5, padding=0)