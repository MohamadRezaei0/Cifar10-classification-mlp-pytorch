import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(1024, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.25)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(F.relu(x))
        x = self.dropout(x)
        x = self.fc2(F.relu(self.bn1(x)))
        x = self.dropout(x)
        x = self.fc3(F.relu(x))
        x = self.sm(x)
        return x

    def train(self, trainloader, optimizer, criterion, epochs):
        # in order to return loss and accuracy
        acc_hist = []
        loss_hist = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            # in order to log accuracy
            acc = []
            total_images = 0
            total_correct = 0
            # in order to log loss
            running_loss = []
            for i, data in enumerate(trainloader, 0):
                # data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # evaluate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_images += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                running_loss.append(loss.item())


            # print loss and accuracy
            mean_loss = np.mean(running_loss)
            model_accuracy = total_correct / total_images * 100
            print('epoch: {0:<4} | loss: {1:<6} | accuracy: {2:<4}'.
            format(epoch + 1, round(mean_loss, 3), round(model_accuracy, 2)))

            loss_hist.append(mean_loss)
            acc_hist.append(model_accuracy)

        print('Finished Training')
        return [loss_hist, acc_hist]


    def score(self, testloader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = self(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    def loss(self, dataloader, criterion):
        loss = []
        with torch.no_grad():
            for data in dataloader:
                pred = self(data[0].to(device)) # predict image label
                output = torch.Tensor.cpu(pred) # bring data to the memory
                loss.append(criterion(output, data[1])) # calculate loss with predictions and true labels

        return np.mean(loss)

    def predict(self, dataloader):
        y_true = []
        outputs  = []
        with torch.no_grad():
            for data in dataloader:
                input, label = data[0].to(device), data[1].to(device)
                output = self(input)
                _ , predicted = torch.max(output.data, 1)
                outputs.append(np.array(torch.Tensor.cpu(predicted)))
                y_true.append(np.array(torch.Tensor.cpu(label)))

        return np.ravel(outputs), y_true
