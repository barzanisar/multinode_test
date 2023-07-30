import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.utils.data.distributed

from accelerate import Accelerator

import argparse

parser = argparse.ArgumentParser(
    description='cifar10 classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')


def main():
    print("Starting...")

    args = parser.parse_args()

    accelerator = Accelerator()

    device = accelerator.device

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    net.to(device)

    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = CIFAR10(root='../data', train=True,
                            download=False, transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(
        train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)

    net, optimizer, train_loader = accelerator.prepare(
        net, optimizer, train_loader)

    for batch in train_loader:

        inputs, targets = batch
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()

        print("Done!")


if __name__ == '__main__':
    main()
