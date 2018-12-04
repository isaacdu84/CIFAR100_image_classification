
# --------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

num_classes = 100
num_epochs = 20
batch_size = 200
learning_rate = 0.001

momentum = 0.9
# --------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------

train_set = torchvision.datasets.CIFAR100(root='./data_cifar', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data_cifar', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
    'bottles', 'bowls', 'cans', 'cups', 'plates',
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
    'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain','sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple', 'oak', 'palm', 'pine', 'willow',
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']

subclasses=['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical devices',
            'household furniture','insects','large carnivores','large man-made outdoor things',
            'large natural outdoor scenes','large omnivores','medium-sized mammmals','non-insect invertebrates',
            'people','reptitles','small mammals','trees','vehicles 1','vehicles 2']
# --------------------------------------------------------------------------------------------
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
plt.show()

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------------------------------------------------------------

class SimpleNet(nn.Module):

    def __init__(self, num_classes=100):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16* 5* 5, 120)# flatern
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))

        out=out.view(-1,16*5*5)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
# --------------------------------------------------------------------------------------------

# Choose the right argument for x                                           # error5: choose x
cnn = SimpleNet(num_classes)
cnn.cuda()
# --------------------------------------------------------------------------------------------
# Choose the right argument for x
criterion = nn.CrossEntropyLoss()                                                  # (use different) Loss function
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)    # (use different) Optimizer
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
loss_1=[]
import time
start = time.clock()

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data
        #images = images.view(-1, 3 * 32 * 32)                    # error6: the dimension of input should be 3*32*32
        images, labels = Variable(images.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_1.append(loss.item())

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.item()))


end = time.clock()
print("time used", end-start)
# --------------------------------------------------------------------------------------------


# _, predicted = torch.max(outputs.data, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
class_correct = list(0. for i in range(100))
class_total = list(0. for i in range(100))
subclass_correct=list(0. for i in range(20))
subclass_total=list(0. for i in range(20))
for data in test_loader:
    images, labels = data
    images = Variable(images.cuda())
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)
    for i in range(200):
        label = labels[i]
        if (label>=0) and (label <5):
            subclass_correct[0]+= c[i]
            subclass_total[0]+= 1

        elif (label>=5) and (label <10):
            subclass_correct[1]+= c[i]
            subclass_total[1]+= 1

        elif (label>=10) and (label <15):
            subclass_correct[2]+= c[i]
            subclass_total[2]+= 1

        elif (label>=15) and (label <20):
            subclass_correct[3]+= c[i]
            subclass_total[3]+= 1

        elif (label>=20) and (label <25):
            subclass_correct[4]+= c[i]
            subclass_total[4]+= 1

        elif (label>=25) and (label <30):
            subclass_correct[5]+= c[i]
            subclass_total[5]+=1

        elif (label>=30) and (label <35):
            subclass_correct[6]+= c[i]
            subclass_total[6]+=1

        elif (label>=35) and (label <40):
            subclass_correct[7]+= c[i]
            subclass_total[7]+=1

        elif (label>=40) and (label <45):
            subclass_correct[8]+= c[i]
            subclass_total[8]+=1

        elif (label>=45) and (label <50):
            subclass_correct[9]+= c[i]
            subclass_total[9]+=1

        elif (label >= 51) and (label < 56):
            subclass_correct[10] += c[i]
            subclass_total[10] += 1

        elif (label>=55) and (label <60):
            subclass_correct[11]+= c[i]
            subclass_total[11]+=1

        elif (label>=60) and (label <65):
            subclass_correct[12]+= c[i]
            subclass_total[12]+=1

        elif (label>=65) and (label <70):
            subclass_correct[13]+= c[i]
            subclass_total[13]+=1

        elif (label>=70) and (label <75):
            subclass_correct[14]+= c[i]
            subclass_total[14]+=1


        elif (label >= 75) and (label < 80):
            subclass_correct[15] += c[i]
            subclass_total[15] += 1

        elif (label >= 80) and (label < 85):
            subclass_correct[16] += c[i]
            subclass_total[16] += 1

        elif (label >= 85) and (label < 90):
            subclass_correct[17] += c[i]
            subclass_total[17] += 1

        elif (label >= 90) and (label < 95):
            subclass_correct[18] += c[i]
            subclass_total[18] += 1

        elif (label >= 95) and (label < 100):
            subclass_correct[19] += c[i]
            subclass_total[19] += 1


        class_correct[label] += c[i]
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(100):
    if class_total[i]!=0:
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

print(' ')
print(' ')

for i in range(20):
    if subclass_total[i]!=0:
        print('Accuracy of %5s : %2d %%' % (subclasses[i], 100 * subclass_correct[i] / subclass_total[i]))

print(' ')
print('Accuracy of subclasses is', sum(subclass_correct)/sum(subclass_total)*100)
print(' ')

# -----------------------------------------------------------------------------------
# Test the Model

cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())                        # error8: the dimension of input
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += torch.sum((predicted == labels)).cpu().numpy()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------
##plot training loss

plt.figure("loss function")
plt.plot(loss_1)
plt.show()

print("test")
a=cnn.conv2.weight
a=a[0][0]
b=a.cpu().detach().numpy()
ker1_0 = np.matrix(b)
plt.figure(5)
plt.imshow(ker1_0, cmap='Greys',  interpolation='nearest')
plt.title('One Kernels for Conv1')
plt.show()




imshow(torchvision.utils.make_grid(a))
plt.show()

print("test")
torch.save(cnn.state_dict(), 'model.pkl')