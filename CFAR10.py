from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import pickle
from PIL import Image
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
import matplotlib.pyplot as plt

# data collection
epoch_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#class MyNet(nn.Module):
    #def __init__(self):
        #super(Net, self).__init__()
        # self.conv3_32 = nn.Conv2d(3, 32, kernel_size=3)
        # self.batch_norm_3 = nn.BatchNorm2d(3)
        # self.conv3_64_1 = nn.Conv2d(3, 64, kernel_size=3)
        # self.conv3_64_2 = nn.Conv2d(64, 64, kernel_size=3)
        # self.batch_norm_64 = nn.BatchNorm2d(64)
        # #self.maxpool_3 = F.max_pool2d(kernel_size=3)
        # self.conv3_128_1 = nn.Conv2d(64, 128, kernel_size=3)
        # self.conv3_128_2 = nn.Conv2d(128, 128, kernel_size=3)
        # self.conv3_256_1 = nn.Conv2d(128, 256, kernel_size=3)
        # self.batch_norm_128 = nn.BatchNorm2d(128)
        # self.conv3_256_2 = nn.Conv2d(256, 256, kernel_size=3)
        # self.batch_norm_256 = nn.BatchNorm2d(256)
        # self.conv3_drop = nn.Dropout2d()
        # self.conv3_512_1 = nn.Conv2d(256, 512, kernel_size=3)
        # self.batch_norm_512 = nn.BatchNorm2d(512)
        # self.conv3_512_2 = nn.Conv2d(512, 512, kernel_size=3)
        # self.conv1_512 = nn.Conv2d(512, 512, kernel_size=3)
        # self.conv3_1024_1 = nn.Conv2d(512, 1024, kernel_size=3)
        # self.conv3_1024_2 = nn.Conv2d(1024, 1024, kernel_size=3)
        # self.batch_norm_1024 = nn.BatchNorm2d(1024)
        # # self.rev_conv_256 = nn.Conv2d(512, 256, kernel_size=3)
        # # self.rev_conv_128 = nn.Conv2d(256, 128, kernel_size=3)
        # # self.rev_conv_64 = nn.Conv2d(128, 64, kernel_size=3)
        # # self.rev_conv_10 = nn.Conv2d(64, 10, kernel_size=3)
        # self.globpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.rev_conv1_10 = nn.Conv2d(1024, 10, kernel_size=1)


 #   def forward(self, x):
        #implement forward network
        # x = F.relu(F.max_pool2d(self.conv3_64_1(x), 3, stride=1))
        # x = F.relu(self.batch_norm_64(self.conv3_64_2(x)))
        # x = F.relu(self.conv3_128_1(x))
        # x = F.relu(self.batch_norm_128(self.conv3_128_2(x)))
        # x = F.relu(self.conv3_256_1(x))
        # x = F.relu(self.batch_norm_256(self.conv3_256_2(x)))
        # x = F.relu(self.batch_norm_256(self.conv3_256_2(x)))
        # x = F.relu(self.batch_norm_512(self.conv3_512_1(x)))
        # x = F.relu(F.max_pool2d(self.batch_norm_512(self.conv3_512_2(x)), 3, stride=1))
        # x = F.relu(self.batch_norm_512(self.conv3_512_2(x)))
        # x = F.relu(self.batch_norm_512(self.conv3_512_2(x)))
        # x = F.relu(self.batch_norm_512(self.conv3_512_2(x)))
        # #x = F.relu(self.batch_norm_1024(self.conv3_1024_1(x)))
        # #x = F.relu(self.batch_norm_1024(self.conv3_1024_2(x)))
        #
        # x = self.globpool(x)
        # x = self.rev_conv1_10(x)
        # x = x.view(-1, 10)
        # return F.log_softmax(x, dim=1)
        # x = F.relu(F.max_pool2d(self.conv3_64_1(x), 3, stride=1))
        # x = F.relu(self.batch_norm_64(self.conv3_64_2(x)))
        # x = F.relu(self.batch_norm_128(self.conv3_128_1(x)))
        # x = F.relu(self.conv3_256_1(x))
        # x = F.relu(self.batch_norm_256(self.conv3_256_2(x)))
        # x = F.relu(self.batch_norm_512(self.conv3_512_1(x)))
        # x = F.relu(F.max_pool2d(self.conv3_512_2(x), 3, stride=1))
        # x = F.relu(self.batch_norm_1024(self.conv3_1024_1(x)))
        # x = self.globpool(x)
        # x = self.rev_conv1_10(x)
        # x = x.view(-1, 10)
  #       return F.log_softmax(x, dim=1)


# class VGG(nn.Module):
#
#     def __init__(self, features, num_classes=10, init_weights=True):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 1024),#512 * 7 * 7
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(1024, num_classes),
#         )
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return F.log_softmax(x, dim=1)
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#
# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
#
# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=10):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         #x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return F.log_softmax(x, dim=1)

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=7, stride=1)
        out = out.view(features.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)

def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model


#kwargs2['init_weights'] = False
net = densenet121().cuda()
net.load_state_dict(torch.load("save/xxx.pwf"))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #push data and target to device
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, records_acc, records_avg_loss, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        records_acc.insert(len(records_acc), 100. * correct / len(test_loader.dataset))
        records_avg_loss.insert(len(records_avg_loss), test_loss)
        if epoch >= 100:
            torch.save(model.state_dict(), "save/xxx_{:d}_{:.4f}_({:.0f}%).pwf".format(epoch, test_loss, 100. * correct / len(test_loader.dataset)))


def adj_learn_step(optimizer, epoch):
    if epoch == 100:
        lr = 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('learning rate adjusted to ', lr)
    # if epoch == 150:
    #     lr = 0.0001
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     print('learning rate adjusted to ', lr)
    # if epoch == 250:
    #     lr = 0.00001
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    #     print('learning rate adjusted to ', lr)


def main():

    accuracy = []
    avg_loss = []

    #argument, optional
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=330, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #print(torch.cuda.is_available())
    #seed
    torch.manual_seed(args.seed)

    #specify if want to use pgu
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #dataset
    train_data = CIFAR10(transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ]))
    test_data = CIFAR10(train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))


    #dataloader
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    ##preprocess for MyNet
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    #
    # testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    #model = Net().to(device)
    #print(model)
    #optimizer = optim.Adam(net.parameters(), lr=0.0001)  # , momentum=args.momentum
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        adj_learn_step(optimizer, epoch)
        train(args, net, device, train_loader, optimizer, epoch)
        test(args, net, device, test_loader, accuracy, avg_loss, epoch)
        print(accuracy)

    print(epoch_num)
    print(accuracy)
    print(avg_loss)


def mean_std(x):
    meanr = 0
    meang = 0
    meanb = 0
    stdr = 0
    stdg = 0
    stdb = 0
    length = 0
    for index, item in enumerate(x):
        meanr += np.mean(item[0, :, :]/255)
        meang += np.mean(item[1, :, :]/255)
        meanb += np.mean(item[2, :, :]/255)
        stdr += np.std(item[0, :, :]/255)
        stdg += np.std(item[1, :, :]/255)
        stdb += np.std(item[2, :, :]/255)
        length = index
    meanr = meanr/length
    meang = meang / length
    meanb = meanb / length
    stdr = stdr / length
    stdg = stdg / length
    stdb = stdb / length
    return meanr, meang, meanb, stdr, stdg, stdb


class CIFAR10(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        'Initialization'
        if self.train:
            path = 'cifar-10-batches-py/'
            file1 = 'data_batch_1'
            file2 = 'data_batch_2'
            file3 = 'data_batch_3'
            file4 = 'data_batch_4'
            file5 = 'data_batch_5'
            self.train_list = []
            self.train_label_list = []
            f1 = open(path+file1, 'rb')
            f2 = open(path + file2, 'rb')
            f3 = open(path + file3, 'rb')
            f4 = open(path + file4, 'rb')
            f5 = open(path + file5, 'rb')
            dict1 = pickle.load(f1, encoding='latin1')
            dict2 = pickle.load(f2, encoding='latin1')
            dict3 = pickle.load(f3, encoding='latin1')
            dict4 = pickle.load(f4, encoding='latin1')
            dict5 = pickle.load(f5, encoding='latin1')
            self.train_list.append(dict1['data'])
            self.train_list.append(dict2['data'])
            self.train_list.append(dict3['data'])
            self.train_list.append(dict4['data'])
            self.train_list.append(dict5['data'])
            self.train_list = np.concatenate(self.train_list, axis=0)
            self.train_label_list.append(dict1['labels'])
            self.train_label_list.append(dict2['labels'])
            self.train_label_list.append(dict3['labels'])
            self.train_label_list.append(dict4['labels'])
            self.train_label_list.append(dict5['labels'])
            self.train_label_list = np.concatenate(self.train_label_list, axis=0)
            self.train_list = np.reshape(self.train_list, (50000, 3, 32, 32))
            self.train_list = self.train_list.transpose((0, 2, 3, 1))
        else:
            path = 'cifar-10-batches-py/'
            file_test = 'test_batch'
            ftest = open(path + file_test, 'rb')
            self.test_list = []
            self.test_label = []
            dict_test = pickle.load(ftest, encoding='latin1')
            self.test_list.append(dict_test['data'])
            self.test_list = np.concatenate(self.test_list, axis=0)
            self.test_label.append(dict_test['labels'])
            self.test_label = np.concatenate(self.test_label, axis=0)
            self.test_list = np.reshape(self.test_list, (10000, 3, 32, 32))
            self.test_list = self.test_list.transpose((0, 2, 3, 1))

    def __len__(self):
        """Denotes the total number of samples"""
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)

    def __getitem__(self, index):
        """Generates one sample of data"""
        if self.train:
            x = self.train_list[index]
            y = self.train_label_list[index]
        else:
            x = self.test_list[index]
            y = self.test_label[index]

        y = np.long(y)
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y



if __name__ == '__main__':
    main()

#side testing
# path = 'cifar-10-batches-py/'
# file1 = 'data_batch_1'
# file2 = 'data_batch_2'
# file3 = 'data_batch_3'
# file4 = 'data_batch_4'
# file5 = 'data_batch_5'
# train_list = []
# train_label_list = []
# f1 = open(path+file1, 'rb')
# f2 = open(path + file2, 'rb')
# f3 = open(path + file3, 'rb')
# f4 = open(path + file4, 'rb')
# f5 = open(path + file5, 'rb')
# dict1 = pickle.load(f1, encoding='bytes')
# dict2 = pickle.load(f2, encoding='bytes')
# dict3 = pickle.load(f3, encoding='bytes')
# dict4 = pickle.load(f4, encoding='bytes')
# dict5 = pickle.load(f5, encoding='bytes')
# train_list.append(dict1[b'data'])
# train_list.append(dict2[b'data'])
# train_list.append(dict3[b'data'])
# train_list.append(dict4[b'data'])
# train_list.append(dict5[b'data'])
# train_list = np.concatenate(train_list, axis=0)
# train_label_list.append(dict1[b'labels'])
# train_label_list.append(dict2[b'labels'])
# train_label_list.append(dict3[b'labels'])
# train_label_list.append(dict4[b'labels'])
# train_label_list.append(dict5[b'labels'])
# train_label_list = np.concatenate(train_list, axis=0)
# train_list = np.reshape(train_list, (50000, 3, 32, 32))
# train_list = train_list.transpose((0, 2, 3, 1))
# #train_list = Image.fromarray(train_list)
# #train_list = transforms.ToTensor()(train_list)
# R, G, B, sR, sG, sB = mean_std(train_list)
# print (R)
# #plt.imshow(train_list[0])
# #plt.show()







