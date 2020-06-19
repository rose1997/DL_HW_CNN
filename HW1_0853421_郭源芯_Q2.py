# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:51:06 2020

@author: Rose Kuo
Deep learning HW1 Q2
"""
import os
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

training_label=pd.read_csv('train.csv')
testing_label=pd.read_csv('test.csv')

directory='C:/Users/fish/Desktop/DL HW1/Q2/images'
train_filename=[filename for filename in training_label['filename']]
test_filename=[filename for filename in testing_label['filename']]
all_filename=[filename for filename in os.listdir(directory)]

#load images and seperate them into training data and testing data
def bounding_box(cat,i):
    xmin = cat.get_value(i,'xmin')
    ymin = cat.get_value(i,'ymin')
    xmax = cat.get_value(i,'xmax')
    ymax = cat.get_value(i,'ymax')
    return (xmin,ymin,xmax,ymax)

train_data=[]
train_label=[]
test_data=[]
test_label=[]
bbox_train=[]
bbox_test=[]
for i in range(len(train_filename)):
    img=cv2.imread("C:/Users/fish/Desktop/DL HW1/Q2/images/"+train_filename[i])
    train_data.append(img)
    train_label.append(training_label.get_value(i,'label'))
    bbox_train.append(bounding_box(training_label,i))
for i in range(len(test_filename)):
    img=cv2.imread("C:/Users/fish/Desktop/DL HW1/Q2/images/"+test_filename[i])
    test_data.append(img)    
    test_label.append(testing_label.get_value(i,'label'))
    bbox_test.append(bounding_box(testing_label,i))
            
#cropping through the bounding box and resize to 80*80
crop_image_train=[]
height = 80
width = 80
dim = (width, height)
for i in range(len(train_data)):
    im=train_data[i]
    if im is not None:
        
        crop_img=im[bbox_train[i][1]:bbox_train[i][3],bbox_train[i][0]:bbox_train[i][2]]
        res = cv2.resize(crop_img, dim, interpolation=cv2.INTER_LINEAR)
        crop_image_train.append(res)
crop_image_test=[]
for i in range(len(test_data)):
    im=test_data[i]
    if im is not None:
        
        crop_img=im[bbox_test[i][1]:bbox_test[i][3],bbox_test[i][0]:bbox_test[i][2]]
        res = cv2.resize(crop_img, dim, interpolation=cv2.INTER_LINEAR)
        crop_image_test.append(res)
#denoise: gaussian blur
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
train_no_noise = []
for i in range(len(crop_image_train)):
    if crop_image_train[i] is None:
        train_no_noise.append(crop_image_train[i])
    else:
        blur = cv2.GaussianBlur(crop_image_train[i], (5, 5), 0)
        train_no_noise.append(blur)
        
test_no_noise = []
for i in range(len(crop_image_test)):
    if crop_image_test[i] is None:
        test_no_noise.append(crop_image_test[i])
    else:
        blur = cv2.GaussianBlur(crop_image_test[i], (5, 5), 0)
        test_no_noise.append(blur)        

#CNN model
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

train=np.empty((len(train_data),80,80,3))
for k in range(len(train_no_noise)):
    train[k,:,:,:]=train_no_noise[k]
train_x = np.einsum('klij->kjli', train)
test=np.empty((len(test_data),80,80,3))
for k in range(len(test_no_noise)):
    test[k,:,:,:]=test_no_noise[k]
test_x = np.einsum('klij->kjli', test)
  
train_y=[]    
count0=0
count1=0
count2=0
for k in range(len(train_label)):
    print(k)
    if train_label[k] == 'bad':
        train_y.append(2)
        count2+=1
    elif train_label[k] == 'none':
        train_y.append(1)
        count1+=1
    elif train_label[k] == 'good':
        train_y.append(0)
        count0+=1
train_y=np.array(train_y)
test_y=[]    
for k in range(len(test_label)):
    print(k)
    if test_label[k] == 'bad':
        test_y.append(2)
    elif test_label[k] == 'none':
        test_y.append(1)
    elif test_label[k] == 'good':
        test_y.append(0)
test_y=np.array(test_y)  
   
EPOCH = 50        
BATCH_SIZE = 50
LR = 0.001 

tensor_data=torch.from_numpy(train_x)
tensor_data=tensor_data.float()
tensor_target=torch.from_numpy(train_y)
tensor_target=tensor_target.long()
torch_dataset = Data.TensorDataset(tensor_data,tensor_target)

test_x=torch.from_numpy(test_x)
test_x=test_x.float()
test_y=torch.from_numpy(test_y)
test_y=test_y.long()

train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 28, 28)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 20 * 20, 3)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization
cnn = CNN()
print(cnn)   

for b_x,b_y in train_loader:
    print(b_x,b_y)
    
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()    
training_loss = []
test_acc =[]
train_acc = []   
correct = 0
def cal_accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return preds, (preds == yb).float().mean()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        output,x = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        #cnn.float()
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            train_pred_y, train_accuracy=cal_accuracy(output,b_y)
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            training_loss.append(loss.data.numpy())
            test_acc.append(accuracy)
            train_acc.append(train_accuracy)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy , '| train accuracy: %.2f' % train_accuracy)

total_train_acc=[]
total_test_acc=[]
total_train_loss=[]
for i in range(len(train_acc)):
    if i % 2 == 1:
        total_train_acc.append(train_acc[i])
        total_test_acc.append(test_acc[i])
        total_train_loss.append(training_loss[i])
        
plt.plot(total_train_acc, label='Training Accuracy')
plt.plot(total_test_acc, label='Testing Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy Rate")
plt.legend(frameon=False)
plt.show()

plt.plot(total_train_loss, label='train loss')
plt.legend(frameon=False)
plt.title("Learning Curve")
plt.xlabel("Number of Epochs")
plt.ylabel("Cross Entropy")
plt.show()

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(test_y, pred_y)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_test.diagonal()

cm_train = confusion_matrix(tensor_target, b_y)
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
cm_train.diagonal()

total_train_acc2=[]
total_test_acc2=[]
total_train_loss2=[]
for i in range(len(train_acc)):
    if i % 2 == 1:
        total_train_acc2.append(train_acc[i])
        total_test_acc2.append(test_acc[i])
        total_train_loss2.append(training_loss[i])
        
plt.plot(total_train_acc2, label='Training Accuracy')
plt.plot(total_test_acc2, label='Testing Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy Rate")
plt.legend(frameon=False)
plt.show()

plt.plot(total_train_loss2, label='train loss')
plt.legend(frameon=False)
plt.title("Learning Curve")
plt.xlabel("Number of Epochs")
plt.ylabel("Cross Entropy")
plt.show()

cm_test = confusion_matrix(test_y, pred_y)
cm_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
cm_test.diagonal()

cm_train = confusion_matrix(tensor_target, train_pred_y)
cm_train = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
cm_train.diagonal()