
# --------------------------------------------------------------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




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
    def __init__(self, input_size, hidden_size, num_classes):    # error3: choose x
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

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data
        images = images.view(-1, 3 * 32 * 32)                    # error6: the dimension of input should be 3*32*32
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.item()))
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
correct = 0
total = 0
label=[]
output=[]
for images, labels in test_loader:
    images = Variable(images.view(-1, 3*32*32).cuda())
    labels = Variable(labels.cuda())
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += torch.sum((predicted == labels)).cpu().numpy()
    label.append(labels)
    output.append(predicted)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# --------------------------------------------------------------------------------------------
#confusion matrix
target = torch.zeros(200, 1,50)
response = torch.zeros(200,1,50)

for i in range(len(label)):
    target[i]=label[i]
    response[i]=output[i]

response=response.data.view(1,-1)
target=target.data.view(1,-1)
response=response.numpy().flatten()
target=target.numpy().flatten()
response_sub=response.tolist()
target_sub=target.tolist()



C_100classes=confusion_matrix(target, response)
print(C_100classes)


for i in range(len(response_sub)):
    if (response_sub[i] >= 0) and (response_sub[i] < 5):
        response_sub[i] = 0

    elif (response_sub[i] >= 5) and (response_sub[i] < 10):
        response_sub[i] = 1

    elif (response_sub[i] >= 10) and (response_sub[i] < 15):
        response_sub[i] = 2

    elif (response_sub[i] >= 15) and (response_sub[i]< 20):
        response_sub[i] = 3

    elif (response_sub[i] >= 20) and (response_sub[i] < 25):
        response_sub[i] = 4

    elif (response_sub[i] >= 25) and (response_sub[i] < 30):
        response_sub[i] = 5

    elif (response_sub[i] >= 30) and (response_sub[i] < 35):
        response_sub[i] = 6

    elif (response_sub[i] >= 35) and (response_sub[i] < 40):
        response_sub[i] = 7

    elif (response_sub[i] >= 40) and (response_sub[i] < 45):
        response_sub[i]=8

    elif (response_sub[i] >= 45) and (response_sub[i] < 50):
        response_sub[i] = 9

    elif (response_sub[i] >= 50) and (response_sub[i] < 55):
        response_sub[i] = 10

    elif (response_sub[i]>= 55) and (response_sub[i] < 60):
        response_sub[i] = 11

    elif (response_sub[i] >= 60) and (response_sub[i] < 65):
        response_sub[i] = 12

    elif (response_sub[i] >= 65) and (response_sub[i] < 70):
        response_sub[i] = 13

    elif (response_sub[i] >= 70) and (response_sub[i] < 75):
        response_sub[i] = 14

    elif (response_sub[i] >= 75) and (response_sub[i] < 80):
        response_sub[i] = 15

    elif (response_sub[i] >= 80) and (response_sub[i] < 85):
        response_sub[i] = 16

    elif (response_sub[i] >= 85) and (response_sub[i]< 90):
        response_sub[i] = 17

    elif (response_sub[i] >= 90) and (response_sub[i]< 95):
        response_sub[i] = 18

    elif (response_sub[i] >= 95) and (response_sub[i]< 100):
        response_sub[i] = 19


for j in range(len(target_sub)):
    if (target_sub[j] >= 0) and (target_sub[j] < 5):
        target_sub[j] = 0

    elif (target_sub[j] >= 5) and (target_sub[j] < 10):
        target_sub[j] = 1

    elif (target_sub[j] >= 10) and (target_sub[j] < 15):
        target_sub[j] = 2

    elif (target_sub[j] >= 15) and (target_sub[j] < 20):
        target_sub[j] = 3

    elif (target_sub[j] >= 20) and (target_sub[j] < 25):
        target_sub[j] = 4

    elif (target_sub[j] >= 25) and (target_sub[j] < 30):
        target_sub[j] = 5

    elif (target_sub[j] >= 30) and (target_sub[j] < 35):
        target_sub[j] = 6

    elif (target_sub[j] >= 35) and (target_sub[j] < 40):
        target_sub[j] = 7

    elif (target_sub[j] >= 40) and (target_sub[j] < 45):
        target_sub[j] = 8

    elif (target_sub[j] >= 45) and (target_sub[j] < 50):
        target_sub[j]=9

    elif (target_sub[j] >= 50) and (target_sub[j] < 55):
        target_sub[j] = 10

    elif (target_sub[j] >= 55) and (target_sub[j] < 60):
        target_sub[j] = 11

    elif (target_sub[j] >= 60) and (target_sub[j] < 65):
        target_sub[j] = 12

    elif (target_sub[j] >= 65) and (target_sub[j] < 70):
        target_sub[j] = 13

    elif (target_sub[j] >= 70) and (target_sub[j] < 75):
        target_sub[j]=14

    elif (target_sub[j] >= 75) and (target_sub[j] < 80):
        target_sub[j] = 15

    elif (target_sub[j] >= 80) and (target_sub[j] < 85):
        target_sub[j] = 16

    elif (target_sub[j] >= 85) and (target_sub[j] < 90):
        target_sub[j] = 17

    elif (target_sub[j] >= 90) and (target_sub[j] < 95):
        target_sub[j] = 18

    elif (target_sub[j] >= 95) and (target_sub[j] < 100):
        target_sub[j] = 19



C_subclasses=confusion_matrix(np.array(target_sub), np.array(response_sub))
#print(C_subclasses)
class_names = subclasses

df_cm = pd.DataFrame(C_subclasses, index=class_names, columns=class_names )

plt.figure(figsize=(14,10))
cm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                 yticklabels=df_cm.columns, xticklabels=df_cm.columns)
cm.yaxis.set_ticklabels(cm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
cm.xaxis.set_ticklabels(cm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
plt.ylabel('Target subclassc label',fontsize=15)
plt.xlabel('Predicted label',fontsize=15)
plt.title('Confusion matrix', fontsize=20)
plt.tight_layout()
plt.show()



torch.save(net.state_dict(), 'model.pkl')