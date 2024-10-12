from PIL import Image

import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)



LAYER = {'resnet20': np.arange(1, (20 - 2) // 2 + 1),  # 9
         'resnet56': np.arange(1, (56 - 2) // 2 + 1),  # 27
         'resnet110': np.arange(2, (110 - 2) // 2 + 1, 2),  # 27
         'wrn40x2': np.arange(1, (40 - 4) // 2 + 1),  # 18
         'wrn28x2': np.arange(1, (28 - 4) // 2 + 1),  # 12
         'wrn16x2': np.arange(1, (16 - 4) // 2 + 1),  # 6
         'resnet34': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet34im': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18im': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet18_': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet34_': np.arange(1, (34 - 2) // 2 + 1),  # 16
         }


def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

    return losses.avg, top1.avg, top5.avg


def train_kl_att(model, model_glb, optimizer, criterion, train_loader, device, args):
    

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion_ce, criterion_kl, criterion_kd = criterion
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            feat_t, output_t,_,__ = model_t(inputs, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
        feat_s, output_s,_,__ = model_s(inputs, is_feat=True)

        loss_ce = criterion_ce(output_s, targets)
        loss_kl = criterion_kl(output_s, output_t)
        loss_kd = criterion_kd(feat_s, feat_t)

        loss = loss_ce + args.alpha * loss_kl + args.beta * loss_kd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output_s, targets, topk=(1, 5))

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

    return losses.avg, top1.avg, top5.avg

# def train_kl(module_list, optimizer, criterion, train_loader, device, args):
#     for module in module_list:
#         module.train()
#     module_list[-1].eval()

#     model_s = module_list[0]
#     model_t = module_list[-1]

#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     criterion_ce, criterion_kl, criterion_kd = criterion
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         with torch.no_grad():
#             feat_t, output_t,_,__ = model_t(inputs, is_feat=True)
#             feat_t = [f.detach() for f in feat_t]
#         feat_s, output_s,_,__ = model_s(inputs, is_feat=True)

#         loss_ce = criterion_ce(output_s, targets)
#         loss_kl = criterion_kl(output_s, output_t)
#         loss_kd = criterion_kd(feat_s, feat_t)

#         loss = loss_ce + args.alpha * loss_kl + args.beta * loss_kd

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         acc1, acc5 = accuracy(output_s, targets, topk=(1, 5))

#         batch_size = targets.size(0)
#         losses.update(loss.item(), batch_size)
#         top1.update(acc1, batch_size)
#         top5.update(acc5, batch_size)

#     return losses.avg, top1.avg, top5.avg


def test(model, test_loader, device):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            batch_size = targets.size(0)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

