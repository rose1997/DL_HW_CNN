# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:53:58 2020

@author: Rose Kuo
Deep learning HW1 Q1
"""
#Q1 Deep Neural Network for Classification
#Q1-1 Please construct a DNN for classification
import numpy as np
from keras.utils import to_categorical
import random
from numpy import newaxis
from datetime import datetime
import matplotlib.pyplot as plt

train_data=np.load('train.npz')
test_data=np.load('test.npz')
test_data1=np.load('test.npz')

for k in train_data.files:
    print(k)
for k in test_data.files:
    print(k)
x_train=train_data['image'].reshape(-1,784)/255
y_train=to_categorical(train_data['label'])
x_test=test_data['image'].reshape(-1,784)/255
y_test=to_categorical(test_data['label'])
x_train = x_train[:, :, newaxis]
y_train = y_train[:, :, newaxis]
x_test = x_test[:, :, newaxis]
y_test = y_test[:, :, newaxis]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#module1 = NN([6, 32, 32, 64, 2])
#module1.SGD(training_data, testing_data, 3000, 100, 0.3)
class DNN():
    def __init__(self,layer_sizes,init):
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        if init=='zero':
            self.weight=[np.zeros((i, j)) for (i, j) in zip(layer_sizes[1:], layer_sizes[:-1])]
            self.bias=np.array([np.zeros((y, 1)) for y in self.layer_sizes[1:]]) 
            print('zero')
        elif init=='random':
            self.weight = [np.random.randn(i, j) for (i, j) in zip(layer_sizes[1:], layer_sizes[:-1])]
            self.biases_initialize()
            print('random')
        self.params = [self.weight, self.bias]
        #print("self.weight[0]"+str(self.weight[0].shape))  #(30,784)
        self.training_loss = []
        self.training_error_rate = []
        self.testing_error_rate = []
        self.f_data0=[]
        self.f_data1=[]
        self.f_data2=[]
        self.f_data3=[]
        self.f_data4=[]
        self.f_data5=[]    
        self.matrics=np.zeros((10,10))
        
    def biases_initialize(self):
        self.bias = np.array([np.random.randn(y, 1) for y in self.layer_sizes[1:]])
            
    def SGD(self,train_data,test_data,min_batch,epoches,lr):
        self.training_loss = []
        self.training_error_rate = []
        self.testing_error_rate = []
        
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], [] 
        
        random.shuffle(train_data)
        min_batches = [train_data[i:i+min_batch] for i in range(0,len(train_data),min_batch)]
        #k=0
        for epoch in range(epoches):
            random.shuffle(train_data)  
            for single_batch in min_batches:
                self.update_parameter(single_batch,lr)
                #print(len(self.weight),len(self.weight[-1]))
            if epoch == 20:
                for x, ground_truth in train_data:
                    #output = self.forward(x)
                    for w,b in zip(self.weight,self.bias):
                        if(w.shape == (10,2)): 
                            self.f_data0.append(x[0])
                            self.f_data1.append(x[1])
                        sigmoid=np.dot(w,x)+b
                        x=1.0/(1.0+np.exp(-sigmoid))
                    self.f_data2.append(x)
                print("epoch 20 done")
            if epoch == 80:
                for x, ground_truth in train_data:
                    #output = self.forward(x)
                    for w,b in zip(self.weight,self.bias):
                        if(w.shape == (10,2)): 
                            self.f_data3.append(x[0])
                            self.f_data4.append(x[1])
                        sigmoid=np.dot(w,x)+b
                        x=1.0/(1.0+np.exp(-sigmoid))
                    self.f_data5.append(x)
                print("epoch 80 done")                  
            if (epoch%50==0):
                #num = self.evaluate(test_data)
                #print "the {0}th epoches: {1}/{2}".format(k,num,len(test_data))
                self.training_loss.append(self.calc_loss(train_data))
                self.training_error_rate.append(self.error(train_data) / len(train_data))
                self.testing_error_rate.append(self.error(test_data) / len(test_data))
                print ('the {0}th epoches:'.format(epoch))
                print('train error rate: %d / %d(%f)' % (self.error(train_data), len(train_data), self.error(train_data) / len(train_data)))
                print('test error rate: %d / %d(%f)' % (self.error(test_data), len(test_data), self.error(test_data) / len(test_data)))                                
                print('training loss: %f' % self.calc_loss(train_data))
                
    def update_parameter(self,single_batch,lr):
        gradient_b = [np.zeros(b.shape) for b in self.bias]
        gradient_w = [np.zeros(w.shape) for w in self.weight]
        for x,y in single_batch:
            b,w = self.backpropagation(x,y)
            gradient_b = [nb +db for nb,db in zip(gradient_b,b)]
            gradient_w = [nw + dw for nw,dw in zip(gradient_w,w)]
        self.bias = [b - lr * gb/len(single_batch) for gb,b in zip(gradient_b,self.bias)]
        self.weight = [w - lr * gw/len(single_batch) for gw,w in zip(gradient_w,self.weight)]

    def forward(self,x):
        #print(self.weight)
        for w,b in zip(self.weight,self.bias):
            sigmoid=np.dot(w,x)+b
            #print(w.shape)
            x=1.0/(1.0+np.exp(-sigmoid))
        return x

    def backpropagation(self,x,y):
        activation = x
        activations = [x]
        zs = []

        for w, b in zip(self.weight, self.bias):
            # print w.shape,activation.shape,b.shape
            z = np.dot(w, activation) +b
            zs.append(z)  
            activation = 1.0 / (1.0 + np.exp(-z))
            # print 'activation',activation.shape
            activations.append(activation)  
        #print(y)
        delta = activations[-1] - y
        delta_w = [np.zeros(w.shape) for w in self.weight]  
        delta_b = [np.zeros(b.shape) for b in self.bias]
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta,activations[-2].T)
        #print('delta_w'+str(delta_w[-1].shape))
        for layer in range(2,self.number_of_layers):
            sigmoid=1.0 / (1.0 + np.exp(-zs[-layer]))
            dsig=sigmoid * (1-sigmoid)
            delta = np.dot(self.weight[-layer+1].T,delta) * dsig
            delta_b[-layer] = delta
            delta_w[-layer] = np.dot(delta,activations[-layer-1].T)
        return (delta_b,delta_w)
    
    def error(self,data):
        self.matrics=np.zeros((10,10))
        compare = [ (np.argmax(self.forward(x)), np.argmax(y)) for x, y in data ]
        error_count=0
        for y1,y2 in compare:
            if y1!=y2:
                error_count=error_count+1
                self.matrics[y1][y2]+=1
            elif y1 == y2:
                self.matrics[y1][y2]+=1
        return error_count   
    
    def calc_loss(self, data):
        loss = 0
        for x, ground_truth in data:
            output = self.forward(x)
            cross_entropy=np.sum( np.nan_to_num( -ground_truth*np.log(output) - (1-ground_truth)*np.log(1-output)))
            loss += cross_entropy / len(data)
        return loss


start1=datetime.now()    
net1 = DNN([784, 30, 10],init='random')
min_batch_size = 100
lr = 0.3
epoches = 3000
train_data = list(zip(x_train, y_train))
test_data = list(zip(x_test, y_test))
net1.SGD(train_data,test_data,min_batch_size,epoches,lr)
print('complete')
print(datetime.now()-start1)

new_x_axis=np.arange(0,3000,50)
fig,ax=plt.subplots(1,1)
ax.plot(new_x_axis,net1.training_loss)
ax.set_title('training loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Avg Cross Entropy')
fig,ax=plt.subplots(1,1)
ax.plot(new_x_axis,net1.training_error_rate)
ax.set_title('training error rate')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error Rate')
fig,ax=plt.subplots(1,1)
ax.plot(new_x_axis,net1.testing_error_rate)
ax.set_title('testing error rate')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error Rate')

#
#Q1-2 Please perform zero and random initializations 
#for the model weights and compare the corresponding error rates.
start2=datetime.now()    
net2 = DNN([784, 30, 10],init='zero')
min_batch_size = 100
lr = 0.3
epoches = 3000
net2.SGD(train_data,test_data,min_batch_size,epoches,lr)
print('complete')
print(datetime.now()-start2)

new_x_axis=np.arange(0,3000,50)
fig,ax=plt.subplots(1,1)
ax.plot(new_x_axis,net2.training_loss)
ax.set_title('training loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Avg Cross Entropy')
fig,ax=plt.subplots(1,1)
ax.plot(new_x_axis,net2.training_error_rate)
ax.set_title('training error rate')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error Rate')
fig,ax=plt.subplots(1,1)
ax.plot(new_x_axis,net2.testing_error_rate)
ax.set_title('testing error rate')
ax.set_xlabel('Epochs')
ax.set_ylabel('Error Rate')


#Q1-3 Design your network architecture
start3=datetime.now()    
net3 = DNN([784, 30, 2, 10],init='random')
min_batch_size = 100
lr = 0.3
epoches = 2000
net3.SGD(train_data,test_data,min_batch_size,epoches,lr)
print('complete')
print(datetime.now()-start3)

f_data=[]
for i in range(len(net3.f_data0)):
    temp=[net3.f_data0[i][0],net3.f_data1[i][0]]
    f_data.append(temp)
    
a=[item[0] for item in f_data] 
b=[item[1] for item in f_data] 

count=[]
for i in range(len(net3.f_data2)):
    count.append(np.argmax(net3.f_data2[i]))
    
plt.figure(figsize=(10, 8))
plt.scatter(a, b, c=count)
plt.colorbar()
plt.show()

f_data1=[]
for i in range(len(net3.f_data3)):
    temp=[net3.f_data3[i][0],net3.f_data4[i][0]]
    f_data1.append(temp)
    
a=[item[0] for item in f_data1] 
b=[item[1] for item in f_data1] 

count=[]
for i in range(len(net3.f_data5)):
    count.append(np.argmax(net3.f_data5[i]))
    
plt.figure(figsize=(10, 8))
plt.scatter(a, b, c=count)
plt.colorbar()
plt.show()
    
#Q1-4 Please list your confusion matrix
from pandas import *
print(DataFrame(net3.matrics))