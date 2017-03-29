import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import mnist


# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                             ])
trainset = mnist.MNIST(root='MNIST', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

testset = mnist.MNIST(root='MNIST', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('0','1', '2', '3', '4', '5', '6', '7', '8', '9')


# functions to show an image
#%matplotlib inline
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

# show some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print labels
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16*5*5)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
#net.cuda()  #cuda

criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2): # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) #cuda
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s'%classes[labels[j]] for j in range(4)))

outputs = net(Variable(images))
#outputs = net(Variable(images).cuda())  #cuda

# the outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class

# So, let's get the index of the highest energy
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s'% classes[predicted[j][0]] for j in range(4)))

correct = 0
total = 0
for data in testloader:
    images, labels = data
    #images, labels = Variable(images.cuda()), Variable(labels.cuda()) #cuda
    #outputs = net(images)  #cuda
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    #correct += (predicted == labels.data).sum()  #cuda
    
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    #images, labels = Variable(images.cuda()), Variable(labels.cuda()) #cuda 
    #outputs = net(images)
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    #c = (predicted == labels.data).squeeze()  #cuda
    for i in range(4):
        label = labels[i]
        #label = labels[i].data #cuda
        #label = label.cpu().numpy()[0]  #cuda
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

#torch.save(net.state_dict(),'/home/wang/git/cifar10/mnist.cuda.weight')  #cuda
torch.save(net.state_dict(),'/home/wang/git/cifar10/mnist.weight')

#model=Net()
#model.cuda() #cuda
#model.load_state_dict(torch.load('/home/wang/git/cifar10/mnist.weight'))
#class_correct = list(0. for i in range(10))
#class_total = list(0. for i in range(10))
#for data in testloader:
#    images, labels = data
#    images, labels = Variable(images.cuda()), Variable(labels.cuda()) #cuda 
#    outputs = model(images)
#    _, predicted = torch.max(outputs.data, 1)
#    #c = (predicted == labels).squeeze()
#    c = (predicted == labels.data).squeeze()  #cuda
#    for i in range(4):
#        #label = labels[i]
#        label = labels[i].data #cuda
#        label = label.cpu().numpy()[0]  #cuda
#        class_correct[label] += c[i]
#        class_total[label] += 1

#for i in range(10):
#    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



