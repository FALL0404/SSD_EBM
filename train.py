from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--input', default=300, type=int, choices=[300, 512],
                    help='SSD input size, currently support ssd300 and ssd512')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO dataset')
parser.add_argument('--dataset_root', default=VOC_ROOT, help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iteration')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data loading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optimizer')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for learning rate')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='/content/weights/', help='Directory for saving checkpoint models')

args = parser.parse_args()


# Set tensor type based on CUDA availability
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("WARNING: CUDA device is available but not used. Set --cuda to True for better performance.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Create save folder if it does not exist
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root for COCO dataset')
            print("WARNING: Using default COCO dataset_root.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root, transform=SSDAugmentation(args.input, MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset_root for VOC dataset')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(args.input, MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    # Build SSD model
    ssd_net = build_ssd('train', args.input, cfg['SSD{}'.format(args.input)]['num_classes'])
    net = ssd_net
    print('Network architecture: ', net)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    # Load weights if resuming, otherwise load base VGG weights
    if args.resume:
        print('Resuming training, loading weights from {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(os.path.join(args.save_folder, args.basenet))
        print('Loading base VGG network weights...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing new layers\' weights...')
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # Set optimizer and loss criterion
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['SSD{}'.format(args.input)]['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()
    loc_loss, conf_loss, epoch = 0, 0, 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the following args:', args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)

    batch_iterator = iter(data_loader)

    # Training loop
    for iteration in range(args.start_iter, cfg['SSD{}'.format(args.input)]['max_iter']):
        if args.visdom and iteration != 0 and iteration % epoch_size == 0:
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, 'append', epoch_size)
            loc_loss, conf_loss = 0, 0
            epoch += 1

        if iteration in cfg['SSD{}'.format(args.input)]['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), requires_grad=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, requires_grad=True) for ann in targets]

        # Forward pass and loss calculation
        t0 = time.time()
        out = net(images)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.time()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print(f'Timer: {t1 - t0:.4f} sec. || Iteration {iteration} || Loss: {loss.item():.4f}')

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(), iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), f'{args.save_folder}/ssd{args.input}_VOC_b32_{iteration}.pth')

    torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'{args.dataset}.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(xlabel, ylabel, title, legend):
    return viz.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1, 3)).cpu(),
                    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title, legend=legend))


def update_vis_plot(iteration, loc, conf, window1, window2, update_type, epoch_size=1):
    viz.line(X=torch.ones((1, 3)).cpu() * iteration, 
             Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
             win=window1, update=update_type)

    if iteration == 0:
        viz.line(X=torch.zeros((1, 3)).cpu(), 
                 Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(), 
                 win=window2, update=True)


if __name__ == '__main__':
    train()
