import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import collections
import pandas as pd
import copy

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(0)

BATCH_SIZE = 92
DATASET = "Cifar"
TRAIN_NAME = "Cifar_full"
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


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

print('val length:', len(image_datasets['valid']))
print('train length:', len(image_datasets['train']))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        model_ft = torchvision.models.mobilenet_v2()
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, 10)
        self.model = model_ft

    def forward(self, x):
        return self.model(x)

def test_model(model, d_loader):
    model.eval()
    running_corrects = 0
    total = 0

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
    model = MobileNet()
    model.to(device)
    model.load_state_dict(copy.deepcopy(weights))
    
    results_valid = test_model(model, valid)
    results_train = test_model(model, train)
    
    return {'train': float(results_train), 'valid': float(results_valid)}

def combine_weights(weights_init, weights_end, step=0.05):
    # print(MobileNet())
    results = {'freq': [], 'train':[], 'valid':[]}
    num_steps = int(1/step)
    print("step_size: {}  | num_of_steps: {}".format(step, num_steps))

    starting_weights = weights_init
    for i in range(num_steps+1):
        freq = i * step
        weights_temp = collections.OrderedDict()

        for k, _ in weights_init.items():
            stupid = False
            for name in ['num_batches_tracked']:
                if name in k.split('.'):
                    stupid = True
                    weights_temp[k] = weights_end[k]
            if not stupid:
                weights_temp[k] = (1-freq) * weights_init[k] + freq * weights_end[k]

            # print(torch.sum(weights_temp[k]-weights_init[k]))
            weights_temp.move_to_end(k)
        results_step = calculate_acc(weights_temp)
        print("freq: {} | acc: {}".format(freq, results_step))
        results['freq'].append(freq)
        for k,v in results_step.items():
            results[k].append(v)

    return results

def save_to_csv(dict, path, filename):
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(path, filename))

def evaluate(save_path):

    weights_init = torch.load(os.path.join(save_path, "initial.pth"))
    weights_val = torch.load(os.path.join(save_path, "best_valid.pth"))
    results_val = combine_weights(weights_init, weights_val)
    save_to_csv(results_val, save_path, "interpolation_val.csv")

    weights_init = torch.load(os.path.join(save_path, "initial.pth"))
    weights_final = torch.load(os.path.join(save_path, "final.pth")).state_dict()
    results_train = combine_weights(weights_init, weights_final)
    save_to_csv(results_train, save_path, "interpolation_final.csv")

if __name__ == "__main__":
    evaluate("./Cifar_full")



