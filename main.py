'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import analysis

def get_current_lr(optimi):
    for g in optimi.param_groups:
        return g["lr"]
# Training
# aswt_force_epochs is the number of epochs that must be performed when the LR is changed
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    reported_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        reported_loss = train_loss/(batch_idx+1)
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print("TrainLoss:", (train_loss/(batch_idx+1)))
        print("TrainAcc:", (100.*correct/total))

    train_acc = correct/total
    return (reported_loss, train_acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    reported_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            reported_loss = test_loss/(batch_idx+1)
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print("TestLoss:", (test_loss/(batch_idx+1)))
            print("TestAcc:", 100.*correct/total)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/lrsched1_ckpt.pth')
        best_acc = acc
    return (reported_loss, acc)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--model", "-m", default="alexnet", type=str)
parser.add_argument("--run", default=1, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_map = {}

run_num = args.run
print(args.model, " running on number ", run_num)
model_map["alexnet"] = AlexNet()
model_map["fc1"] = FC1()
model_map["fc2"]= FC2()
model_map["GoogLeNet"] = GoogLeNet()
model_map["lenet"] = LeNet()
model_map["resnet34"] = ResNet34()
model_map["resnet50"] = ResNet50()
model_map["resnet101"] = ResNet101()
model_map["vgg11"] = VGG("VGG11")
model_map["vgg16"] = VGG("VGG16")
model_map["vgg19"] = VGG("VGG19")
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
#net = RegNetX_200MF()
net = model_map[args.model]
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0, weight_decay=0)

log_name = "losses/" + args.model + "_" + str(run_num) + ".txt" 

log_file = open(log_name, "w")
###
# Log Format:
# Epoch, Train Loss, Train Acc, Test Loss, Test Acc
###
test_acc_history = []
aswt_force_val = 0
curr_lr = get_current_lr(optimizer)
gamma = 0.6
count = 5
num_data = 17
local_maxima = 0
slack_prop = 0.05
aswt_start_epoch = min(num_data, count)
for epoch in range(start_epoch, start_epoch+400):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    test_acc_history.append(test_acc)
    # perform ASWT to reduce LR
    if aswt_force_val == 0 and epoch > aswt_start_epoch:
        # aswt test
        aswt_stop = analysis.aswt_stopping(test_acc_history, gamma=gamma, count=count, num_data=num_data, local_maxima=local_maxima, slack_prop=slack_prop)
        if aswt_stop:
            curr_lr = curr_lr * 0.1
            for g in optimizer.param_groups:
                g["lr"] = curr_lr
            aswt_force_val = 5
            print("LR is now", curr_lr)
    else:
        aswt_force_val -= 1
    log_line = str(epoch) + "," + str(train_loss) + "," + str(train_acc) + "," + str(test_loss) + "," + str(test_acc) + "\n"
    log_file.write(log_line) 

log_file.close()
