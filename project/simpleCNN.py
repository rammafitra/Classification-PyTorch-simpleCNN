# IMPORT LIBABRY
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms, datasets


# SET DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# LOAD DATASETS
train_data_directory = "data/train/"
test_data_directory = "data/test"
image_size = 150

train_data_transforms = transforms.Compose([
    transforms.CenterCrop(image_size),
    transforms.RandomRotation(5), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
test_data_transforms =  transforms.Compose([
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_image = datasets.ImageFolder(train_data_directory, transform = train_data_transforms)
test_image = datasets.ImageFolder(test_data_directory, transform=test_data_transforms)
n_labels = len(train_image.classes)

def imshow(data_image, tensor=False):
    image = data_image[0]
    label = data_image[1]
    image = image.numpy().transpose((1,2,0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5],)
    image = std * image + mean
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.show()
    
    print(train_image.classes[label])
 
# CREATE THE LOADER
train_data_loaders = torch.utils.data.DataLoader(train_image, batch_size = 64, shuffle = True)
test_data_loaders = torch.utils.data.DataLoader(test_image, batch_size = 64, shuffle = False)

for image, label, in train_data_loaders:
    for i in range(3):
        imshow((image[i], label[i]))
    break

#CREATE THE ARCHITECTURE
class simpleCNN(nn.Module):
    def __init__(self, ks = 4, ps = 3, fm1 = 16, fm2 = 32, n = 256):
        super(simpleCNN,self).__init__()

        self.conv1 = nn.Conv2d(3, fm1, kernel_size = ks, stride = 1, padding = 0)
        self.pool = nn.MaxPool2d(kernel_size = ps, stride = 2, padding = 0)
        self.conv2 = nn.Conv2d(fm1, fm2, kernel_size = ks, stride = 1, padding = 0)

        # calculate CNN's output size
        result = self.conv_size(self.conv_size(self.conv_size(self.conv_size(image_size,ks),ps ,s=2), ks),ps, s=2)**2*fm2
        self.fc1 = nn.Linear(result, n)
        self.fc2 = nn.Linear(n, n_labels)
        self.do = nn.Dropout()

    def conv_size(self, inp, k, p =0, s = 1):
            return (inp-k+2*p)//s+1

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.do(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = simpleCNN()
model.to(device)

#OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
epoch = 5

#TRAIN
model.train()
for i in range(epoch):
    total_loss = 0
    total_sample = 0
    total_correct = 0

    for image, label in train_data_loaders:
        image = image.to(device)
        label = label.to(device)

        out = model(image)
        loss = criterion(out, label)
        total_loss += loss.item()
        total_sample += len(label)
        total_correct += torch.sum(torch.sum(out,1)[1]==label).item()*1.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch", i, total_loss/total_sample, total_correct/total_sample)

#EVALUATION
# model.eval()
total_loss = 0
total_sample = 0
total_correct = 0

for image, label in train_data_loaders:
        image = image.to(device)
        label = label.to(device)

        out = model(image)
        loss = criterion(out, label)
        total_loss += loss.item()
        total_sample += len(label)
        total_correct += torch.sum(torch.sum(out,1)[1]==label).item()*1.0

print("test loss", total_loss/total_sample)
print("test accuracy", total_correct/total_sample)

#Sanity Check
imagePath = "data/test/JUICE/JUICE0041.png"
imageTest = Image.open(imagePath)
plt.imshow(imageTest)
plt.show()

image_transformed = test_data_transforms(imageTest)
image_transformed = image_transformed.unsqueeze(0).to(device)
out = model(image_transformed)
print(out)
print("PREDICTION", train_image.classes[torch.max(out,1)[1]])


