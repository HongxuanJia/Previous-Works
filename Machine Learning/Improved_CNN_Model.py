from __future__ import print_function
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def dataset():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]) #(0.1307,), (0.3081,)
    trainset = datasets.MNIST('./Data/MNIST_data',download=True,train=True, transform=transform)
    testset = datasets.MNIST('./Data/MNIST_data',train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader =  torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    # 938 * 64  = 60032
    for images ,labels in trainloader:
        dataiter = iter(trainloader)
        images,labels = dataiter.__next__()
        # print(type(images))
        # # print(images.shape)
        # # print(labels.shape)
        # print(images)
        # print(labels)
    json = {
        "trainloader": trainloader,
        "testloader": testloader
    }
    return json

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    plt.figure()
    pic = None
    for batch_idx, (data,target) in enumerate(train_loader):  #给每个(image,label)一个索引值
        # if batch_idx in (1,2,3,4,5):
        #     pic = data[0,0,:,:]
        # else:
        #     pic = torch.cat((pic, data[0, 0, :, :]), dim=1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # cross_entropy equals to:
        # soft_out = F.softmax(out)
        # log_soft_out = torch.log(soft_out)
        # loss = F.nll_loss(log_soft_out, y)

        # gradient
        loss.backward()
        # Optimize the parameters according to the calculated gradients
        optimizer.step()
        #
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    torch.save(model, "./mnist_cnn.pt")

def as_float(x):
     y='{:.3f}'.format(x)
     return(y)

def predict(model,device,predict_loader):
    model.eval()
    with torch.no_grad():
        for images, target in predict_loader:
            predictions = []
            confidence_values = []
            # excel = images[0].numpy()
            # a_pd = pd.DataFrame(excel.squeeze())
            # writer = pd.ExcelWriter('./0.xlsx')
            # # write in ro file, 'sheet1' is the page title, float_format is the accuracy of data
            # a_pd.to_excel(writer, 'sheet1', float_format='%.2f')
            # writer.save()
            # print(target)
            # Using CNN trained before to classified images
            output = model(images)
            # Calculating the confidence value by softmax
            _probability = F.softmax(output,dim=1)
            # Prediction
            output.numpy()
            for i in output:
                prediction = np.argmax(i)
                prediction = int(prediction)
                predictions.append(prediction)
            # print(predictions)
            # Confidence value
            for j in _probability:
                j = j.numpy()
                probability = j.tolist()
                for i in range(len(probability)):
                    probability[i] = as_float(probability[i])
                confidence_values.append(probability)
            print(confidence_values)
            # images.size 查看张量size    images[0].shape 查看第一个纬度数组的shape
            # print(images.size())
            # torch.Size([64, 1, 28, 28])
            images = np.array(images)
            num = len(images)
            for z in range(0,63):
                image = images[z].squeeze()
                plt.gcf().subplots_adjust(top=0.8)
                plt.imshow(image,cmap='gray')
                plt.title("Prediction:{} \n Confidence Value: \n {}\n{}".format(predictions[z],
                                        confidence_values[z][0:5],confidence_values[z][5:10]))
                plt.show()
            break




            # print(probability)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    # learning rate
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # batch_size is a crucial hyper-parameter
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        # Adjust num worker and pin memory according to your computer performance
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    # Make train dataset split
    traindata = dataset()["trainloader"]
    # Make test dataset split

    testdata = dataset()["testloader"]

    # Convert the dataset to dataloader, including train_kwargs and test_kwargs

    # Put the model on the GPU or CPU
    model = CNN().to(device)

    # Create optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Create a schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Begin training and testing
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, traindata, optimizer, epoch)
        test(model, device, testdata)
        scheduler.step()

    # Save the model
    if args.save_model:
        torch.save(model, "./mnist_cnn.pt")

# def train(args, model, device, train_loader, optimizer, epoch):


if __name__ == '__main__':
    main()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.load("./mnist_cnn.pt")
    # predict_loader = dataset()["testloader"]
    # predict(model, device, predict_loader)