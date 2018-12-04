import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd

input_size = 3 * 32 * 32
hidden_size = 500
num_classes = 100
num_epochs = 50
batch_size = 50
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

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
# --------------------------------------------------------------------------------------------

net = Net(input_size, hidden_size, num_classes)
net.cuda()
# --------------------------------------------------------------------------------------------

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
# --------------------------------------------------------------------------------------------

error=[]
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data
        images = images.view(-1, 3 * 32 * 32)
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        error.append(loss.item())

        if (i + 1) % 20 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.item()))
# --------------------------------------------------------------------------------------------


# _, predicted = torch.max(outputs.data, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# --------------------------------------------------------------------------------------------

class_correct = list(0. for i in range(100))
class_total = list(0. for i in range(100))
subclass_correct=list(0. for i in range(20))
subclass_total=list(0. for i in range(20))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1,3* 32 * 32)).cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)
    for i in range(50):
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

        elif (label >= 50) and (label < 55):
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
totalclass_acc=[]
for i in range(100):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        totalclass_acc.append(float(class_correct[i] / class_total[i] * 100))


print(' ')
print(' ')

subclasses_acc=[]
for i in range(20):
    print('Accuracy of %5s : %2d %%' % (subclasses[i], 100 * subclass_correct[i] / subclass_total[i]))
    subclasses_acc.append(float(subclass_correct[i] / subclass_total[i] * 100))



print(' ')
print('Accuracy of subclasses is', sum(subclass_correct)/sum(subclass_total)*100)
print(' ')
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 3*32*32).cuda())
    labels = Variable(labels.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += torch.sum((predicted == labels)).cpu().numpy()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')


#ACC dataframe for plotting later
sub_acc = pd.DataFrame(columns=['Subclasses','ACC'])
sub_acc['Subclasses'] = subclasses
sub_acc['ACC'] = subclasses_acc
sub_acc['ACC']=sub_acc['ACC'].astype(float)
top10sub_acc=sub_acc.sort_values('ACC', ascending=False).head(10)

#top10sub_acc.to_excel('top10sub_acc.xls')


total_acc = pd.DataFrame(columns=['Classes','ACC'])
total_acc['Classes'] = classes
total_acc['ACC'] = totalclass_acc
total_acc['ACC'] = total_acc['ACC'].astype(float)
top10total_acc=total_acc.sort_values('ACC', ascending=False).head(10)
#top10total_acc.ACC=pd.to_numeric(top10total_acc.ACC)

#ACC plot

a = top10sub_acc
a.plot(kind='bar', figsize=(15,10),fontsize=12, color='cornflowerblue',title='Top 10 ACC in subclasses ', stacked=False)
plt.ylabel('ACC %')
plt.title('Top 10 ACC in subclasses')
plt.xticks(range(0, 10), top10sub_acc['Subclasses'])
plt.show()

b = top10total_acc
b.plot(kind='bar', figsize=(15,10),fontsize=12,color='cornflowerblue',title='Top 10 ACC in subclasses ', stacked=False)
plt.xticks(range(0, 10), top10total_acc['Classes'])
plt.title('Top 10 ACC in all classes')
plt.ylabel('ACC %')
plt.show()