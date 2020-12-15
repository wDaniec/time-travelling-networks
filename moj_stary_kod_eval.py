import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import math
import time
import os
import collections
import pandas as pd
import copy
import random

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(0)

BATCH_SIZE = 92
DATASET = "Cifar"
TRAIN_NAME = "crazy"
OUT_SIZE = 10

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_datasets = {'train': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=True, download=True, transform=transform), 
                    'valid': torchvision.datasets.CIFAR10(root='./data_dir_cifar', train=False, download=True, transform=transform)}
 

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid']}

random.seed(0)
temp = int(0.2*len(image_datasets["train"]))
image_datasets["train"].targets[:temp] = [random.randint(0,9) for _ in range(temp)]

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

print('val length:', len(image_datasets['valid']))
print('train length:', len(image_datasets['train']))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def warm_start(model):
    for idx, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if idx > 150:
            continue
        model(inputs)

def test_model(model, d_loader):
    model.train()
    running_corrects = 0
    total = 0

    warm_start(model)
    # Iterate over data.
    for inputs, labels in d_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total += len(preds)

    epoch_acc = running_corrects.double() / total
    return epoch_acc

def calculate_acc(weights):
    train = dataloaders['train']
    valid = dataloaders['valid']
    model = MobileNetV2(10)
    model.to(device)
    model.load_state_dict(copy.deepcopy(weights))
    
    results_valid = test_model(model, valid)
    results_train = test_model(model, train)
    
    return {'train': float(results_train), 'valid': float(results_valid)}

def combine_weights(weights_init, weights_end, step=0.1):
    # print(MobileNet())
    results = {'freq': [], 'train':[], 'valid':[]}
    num_steps = int(1/step)
    print("step_size: {}  | num_of_steps: {}".format(step, num_steps))

    starting_weights = weights_init
    for i in range(num_steps+1):
        freq = i * step
        weights_temp = collections.OrderedDict()

        for k, _ in weights_init.items():
            # for name in ['num_batches_tracked']:
            #     if name in k.split('.'):
            #         stupid = True
            #         weights_temp[k] = weights_end[k]
            weights_temp[k] = (1-freq) * weights_init[k] + freq * weights_end[k]

            # print(torch.sum(weights_temp[k]-weights_init[k]))
            weights_temp.move_to_end(k)
        results_step = calculate_acc(weights_temp)
        print("freq: {} | acc: {}".format(freq, results_step))
        results['freq'].append(freq)
        for k,v in results_step.items():
            results[k].append(v)

    return results

def get_interpolation(weights_init, weights_end, freq):
    weights_temp = collections.OrderedDict()
    for k, _ in weights_init.items():
        weights_temp[k] = (1-freq) * weights_init[k] + freq * weights_end[k]
        weights_temp.move_to_end(k)
    return weights_temp

def combine_weights_AB(weights_init, weights_final, weights_val, step1=1, step2=0.1):
    num_steps1 = int(1/step1)
    num_steps2 = int(1/step2)
    results = {'freq1': [], 'freq2':[], 'train':[], 'valid':[]}

    print("step_size: {} | num_of_steps: {}".format(step1, num_steps1))
    
    for i in range(1, num_steps1+1):
        freq1 = i*step1
        weights_temp_final = get_interpolation(weights_init, weights_final, freq=freq1)
        results_step = calculate_acc(weights_temp_final)
        print("Going to Final. freq : {} | acc: {}".format(i*step1, results_step))

        weights_temp_val = get_interpolation(weights_init, weights_val, freq=freq1)
        results_step = calculate_acc(weights_temp_val)
        print("Going to Best Val. freq: {} | acc: {}".format(i*step1, results_step))

        for j in range(1, num_steps2+1):
            freq2 = j*step2
            weights_temp2 = get_interpolation(weights_temp_final, weights_temp_val, freq=freq2)
            results_step = calculate_acc(weights_temp2)
            print("freq1: {} | freq2: {} | acc: {}".format(freq1, freq2, results_step))
            results['freq1'].append(freq1)
            results['freq2'].append(freq2)
            for k, v in results_step.items():
                results[k].append(v)
    return results
    



def save_to_csv(dict, path, filename):
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(path, filename))

def evaluate(save_path):

    weights_init = torch.load(os.path.join(save_path, "initial.pth"))
    weights_val = torch.load(os.path.join(save_path, "best_early_stop.pth"))
    weights_final = torch.load(os.path.join(save_path, "final.pth"))

    result_last = combine_weights(weights_final, weights_val)
    save_to_csv(result_last, save_path, "interpolation_AB_last.csv")
    # results_total = combine_weights_AB(weights_init, weights_final, weights_val)
    # save_to_csv(results_total, save_path, "interpolation_AB.csv")
    # results_val = combine_weights_AB(weights_init, weights_val)
    # save_to_csv(results_val, save_path, "interpolation_val.csv")
    # results_train = combine_weights_AB(weights_init, weights_final)
    # save_to_csv(results_train, save_path, "interpolation_final.csv")

if __name__ == "__main__":
    evaluate(TRAIN_NAME)



