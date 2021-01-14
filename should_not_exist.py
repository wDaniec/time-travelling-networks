import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import random
import time
import os
import math
import copy
import neptune
import sys
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(0)

BATCH_SIZE = 64
DATASET = "Cifar"
TRAIN_NAME = "warm_start"
# PATH = "./lilImageNet/best_model_199.pth"
SEND_NEPTUNE = True
OUT_SIZE = 10
CIFAR_FACTOR = 1
PATIENCE = 0
NUM_EPOCHS = 150
WEIGHT_DECAY = 0.00004
MOMENTUM = 0.9
LEARNING_RATE = 0.2
MILESTONES = [30, 70, 110]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if SEND_NEPTUNE:
    neptune.init('andrzejzdobywca/pretrainingpp')
    neptune.create_experiment(name=TRAIN_NAME)

def setup_half_loaders():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if DATASET == "Cifar":
        image_datasets = {'train': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=True, download=True, transform=transform), 
                    'val': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=False, download=True, transform=transform)}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join('~/data/lilImageNet', x),
                                            transform=transform)
                    for x in ['train', 'val']}
    full_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    half_sizes = {'train': int(0.5*full_sizes['train']), 'val': full_sizes['val']}

    subset_indices = torch.randperm(full_sizes['train'])[:half_sizes['train']]

    half_dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=4, sampler=SubsetRandomSampler(subset_indices))}
    half_dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=4)

    return half_dataloaders, half_sizes

def setup_full_loaders():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if DATASET == "Cifar":
        image_datasets = {'train': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=True, download=True, transform=transform), 
                    'val': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=False, download=True, transform=transform)}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join('~/data/lilImageNet', x),
                                            transform=transform)
                    for x in ['train', 'val']}
    full_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    full_dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    

    return full_dataloaders, full_sizes

# device = torch.device("cpu")

def setup():
    folder_name = "{}".format(TRAIN_NAME)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def train_phase1(model, criterion, optimizer, scheduler, t_dataloaders, t_sizes, num_epochs=25):
    torch.save(model.state_dict(), "./{}/initial.pth".format(TRAIN_NAME))
    since = time.time()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels) in enumerate(t_dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / t_sizes[phase]
            epoch_acc = running_corrects.double() / t_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if SEND_NEPTUNE:
                neptune.send_metric('{}_loss'.format(phase), epoch_loss)
                neptune.send_metric('{}_acc'.format(phase), epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                print("best epoch in general", epoch)
                best_acc = epoch_acc
                best_epoch = epoch
                torch.save(model.state_dict(), "./{}/best_valid_phase1.pth".format(TRAIN_NAME))
        torch.save(model.state_dict(), "./{}/{}.pth".format(TRAIN_NAME, "final_phase1"))
        print(time.time() - start)
        print()

    time_elapsed = time.time() - since
    print('Training phase 1 completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print("best epoch: ", best_epoch)
    return model

def train_phase2(model, criterion, optimizer, scheduler, t_dataloaders, t_sizes, num_epochs=25):
    torch.save(model.state_dict(), "./{}/initial.pth".format(TRAIN_NAME))
    since = time.time()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for idx, (inputs, labels) in enumerate(t_dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / t_sizes[phase]
            epoch_acc = running_corrects.double() / t_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if SEND_NEPTUNE:
                neptune.send_metric('{}_loss'.format(phase), epoch_loss)
                neptune.send_metric('{}_acc'.format(phase), epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                print("best epoch in general", epoch)
                best_acc = epoch_acc
                best_epoch = epoch
                torch.save(model.state_dict(), "./{}/best_valid_phase2.pth".format(TRAIN_NAME))
        torch.save(model.state_dict(), "./{}/{}.pth".format(TRAIN_NAME, "final_phase2"))
        print(time.time() - start)
        print()

    time_elapsed = time.time() - since
    print('Training phase 2 completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print("best epoch: ", best_epoch)
    return model



class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t = 6, downsample = False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        
        # for main path:
        c  = t * input_channel
        # 1x1   point wise conv
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)
        

    def forward(self, inputs):
        # main path
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace = True)
        x = self.bn3(self.conv3(x))

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x

class MobileNetV2(nn.Module):
    def __init__(self, output_size, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),
            BaseBlock(16, 24, downsample = False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample = False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample = True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample = False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample = True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample = False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):

        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


model_ft = MobileNetV2(10)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum = MOMENTUM)
exp_lr_scheduler1 = lr_scheduler.MultiStepLR(optimizer_ft, milestones=MILESTONES, gamma=0.1)
exp_lr_scheduler2 = lr_scheduler.MultiStepLR(optimizer_ft, milestones=MILESTONES, gamma=0.1)

# exp_lr_scheduler jest wykomentowany!
setup()
half_dataloaders, half_sizes = setup_half_loaders()
full_dataloaders, full_sizes = setup_full_loaders()
model_ft = train_phase1(model_ft, criterion, optimizer_ft, exp_lr_scheduler1, half_dataloaders, half_sizes, num_epochs=NUM_EPOCHS)
model_ft = train_phase2(model_ft, criterion, optimizer_ft, exp_lr_scheduler2, full_dataloaders, full_sizes, num_epochs=NUM_EPOCHS)

if SEND_NEPTUNE:
    neptune.stop()
    sys.exit()

