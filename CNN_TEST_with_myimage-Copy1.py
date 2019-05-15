#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn import datasets, model_selection

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2

#만들어 놓은 collect_image 함수를 쓰기위해 import
import Image_pick as ip
#미리 만들어 놓은 신경망 클래스 호출
from AnnClass import Net

#mnist 손글씨 이미지 데이터 호출
mnist = datasets.fetch_mldata('MNIST Original', data_home = './data')

#목적변수를 변수에 할당
mnist_label = mnist.target
#70000행, 784 열 정규화한 data
mnist_data = mnist.data/255
#신경망 클래스 설정 
model = Net()


# In[2]:


#불러올 이미지의 경로와 이름
filePath = "./data/811.jpg"
data = []
data = ip.collect_image (filePath, 12, 172, 100, 100)


# In[3]:


#방금 불러온 나의 손글씨 데이터를 추가
data=np.array(data)
data=data.reshape(len(data),784,)
mnist_data=np.vstack((mnist_data,data))
print(mnist_data.shape)
label_extend = np.array([7,9,8,4,2,5,6,3,1,0,7,9,8,6,3,0,2,5,1,4,9,7,8,4,0,6,2,3,5,1,9,7,8,6,5,4,0,3,2,1,
                            9,7,8,6,3,0,1,2,5,4,9,8,7,0,1,6,3,2,5,4,9,8,7,6,5,4,2,0,3,9,1,8,7,6,4,5,2,0,3,9,
                            1,8,7,6,5,4,2,3,0,1,9,8,7,6,4,5,3,2,0,1,8,9,7,6,5,4,3,2,1,0,7,6,8,9,4,5,3,2,0,1,
                            6,4,5,7,9,3,2,8,0,1,9,7,5,4,6,8,3,2,0,1,9,7,4,8,5,6,3,2,1,0,9,2,8,7,0,4,1,6,3,5,
                            7,9,8,5,6,4,3,0,2,1])
mnist_label=np.hstack((mnist_label,label_extend))
print(mnist_label.shape)


# In[4]:


#불러올 이미지의 경로와 이름
filePath = "./data/666.jpg"
data = ip.collect_image (filePath, 12, 150, 100, 100)


# In[5]:


#방금 불러온 나의 손글씨 데이터를 추가
data=np.array(data)
data=data.reshape(len(data),784,)
mnist_data=np.vstack((mnist_data,data))
print(mnist_data.shape)
label_extend = np.array([6]*len(data))
mnist_label=np.hstack((mnist_label,label_extend))
print(mnist_label.shape)


# In[6]:


filePath = "./data/999.jpg"
data = ip.collect_image (filePath, 12, 124, 100, 100)


# In[7]:


#방금 불러온 나의 손글씨 데이터를 추가
data=np.array(data)
data=data.reshape(len(data),784,)
mnist_data=np.vstack((mnist_data,data))
print(mnist_data.shape)
label_extend = np.array([9]*len(data))
mnist_label=np.hstack((mnist_label,label_extend))
print(mnist_label.shape)


# In[8]:


filePath = "./data/000.jpg"
data = ip.collect_image (filePath, 12, 124, 100, 100)


# In[9]:


#방금 불러온 나의 손글씨 데이터를 추가
data=np.array(data)
data=data.reshape(len(data),784,)
mnist_data=np.vstack((mnist_data,data))
print(mnist_data.shape)
label_extend = np.array([0]*len(data))
mnist_label=np.hstack((mnist_label,label_extend))
print(mnist_label.shape)


# In[13]:


train_size = 15000 #훈련 데이터 건수
test_size = 1000 #테스트 데이터 건수
#데이터 집합을 훈련과 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(mnist_data,mnist_label,train_size=train_size,test_size=test_size)


# In[14]:


train_X = train_X.reshape((len(train_X),1,28,28))
test_X = test_X.reshape((len(test_X),1,28,28))


# In[15]:


#준비가 끝난 데이터를 파이토치가 다룰 수 있는 형태로 정리 tensor생성
#훈련 데이터 텐서 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()
#데스트 데이터 텐서 변환
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()
print(train_X.shape)


# In[16]:


#설명변수와 목적변수 텐서를 합침, 총 train 데이터쌍을 만들기 위해서
train = TensorDataset(train_X, train_Y)
#설명변수 784 개 목적변수 1개
#print(train[0])
#미니배치 분할, 미니배치 학습을 위해 데이터를 셔플링해서 100건 단위로 분할
train_loader = DataLoader(train, batch_size = 100, shuffle =True)


# In[17]:


#모형학습
#오차함수 객체
criterion = nn.CrossEntropyLoss()
#최적화를 담당할 객체
optimizer = optim.SGD(model.parameters(),lr=0.01)
#학습시작
for epoch in range(1000):
    total_loss = 0
    #분할해 둔 데이터 꺼내오기
    for train_x, train_y in train_loader:
        #계산 그래프 구성
        train_x, train_y = variable(train_x),variable(train_y)
        #경사초기화
        optimizer.zero_grad()
        #순전파 계산
        output = model(train_x)
        #오차계산
        loss = criterion(output,train_y)
        #역전파계산
        loss.backward()
        #가중치 업데이트
        optimizer.step()
        #누적 오차 계산
        total_loss+=loss.item()
    if(epoch + 1)%100 == 0:
        print(epoch+1, total_loss)
        #100회 반복마다 누적 오차 출력


# In[18]:


#100개씩 분할한 batch를 1000번 반복하여 
#인공지능을 훈련시켰다.
#계산 그래프 구성
test_x , test_y = variable(test_X),variable(test_Y)
#출력이 0 또는 1이 되게 함
result = torch.max(model(test_x).data,1)[1]
#모형의 정확도 측정
#softmax로 출력된 배열에서 제일 확률 높은 곳의 숫자를 추출하고
#그 추출된 결과가 해당 목적변수와 같은지를 판단하고
#같은 갯수와 전체 데이터의 갯수의 비율로 오차를 알아낸다.
accuracy = sum(test_y.data.numpy()==result.numpy())/len(test_y.data.numpy())


# In[19]:


print(accuracy)


# In[21]:


#학습시킨 데이터를 cnn_test_model4라는 이름으로 저장
savePath = "./data/cnn_test_model4.pth"
torch.save(model.state_dict(), savePath)


# In[22]:


#앞서 저장한 학습시킨 데이터를 끌어옴
new_model = Net()
new_model.load_state_dict(torch.load(savePath))


# In[ ]:




