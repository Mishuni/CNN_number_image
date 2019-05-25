#!/usr/bin/env python
# coding: utf-8

# In[83]:


import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn import model_selection

import pandas as pd
from random import *

from matplotlib import pyplot as plt
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2
import pickle
#만들어 놓은 collect_image 함수를 쓰기위해 import
import Extract_number as en
#미리 만들어 놓은 신경망 클래스 호출
from AnnClass import Net

data = []
label = []
#신경망 클래스 설정 
model = Net()

#이미지 data들 불러오기
with open("number_data.txt", "rb") as fp:
    data = pickle.load(fp)
print(data.shape)

#목적변수 불러오기
with open("number_label.txt", "rb") as fp:
    label = pickle.load(fp)
print(label.shape)


# In[84]:


#data 중 약 30000개의 이미지를 rotation 시킨다.
rows, cols = (28,28)
for count in range(1,30000):
    i = randint(0,70360)
    i2 = randint(-90,90)
    n = data[i].reshape(28,28)
    # 이미지의 중심점을 기준으로 90도 회전 하면서 0.5배 Scale
    M= cv2.getRotationMatrix2D((cols/2, rows/2),i2,1)
    dst = cv2.warpAffine(n, M,(cols, rows))
    if(count%3000==0):
        plt.figure(figsize=(2,2))
        plt.imshow(dst,cmap='gray')
    dst = dst / 255 ;
    data[i]=dst.reshape(1,784);


# In[86]:


train_size = 15000 #훈련 데이터 건수
test_size = 1000 #테스트 데이터 건수
#데이터 집합을 훈련과 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data,label,train_size=train_size,test_size=test_size)

train_X = train_X.reshape((len(train_X),1,28,28))
test_X = test_X.reshape((len(test_X),1,28,28))

#준비가 끝난 데이터를 파이토치가 다룰 수 있는 형태로 정리 tensor생성
#훈련 데이터 텐서 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()
#데스트 데이터 텐서 변환
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()
print(train_X.shape)


# In[87]:


#설명변수와 목적변수 텐서를 합침, 총 train 데이터쌍을 만들기 위해서
train = TensorDataset(train_X, train_Y)
#설명변수 784 개 목적변수 1개
#print(train[0])
#미니배치 분할, 미니배치 학습을 위해 데이터를 셔플링해서 100건 단위로 분할
train_loader = DataLoader(train, batch_size = 100, shuffle =True)


# In[88]:


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


# In[90]:


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
i=0
for k in test_y.data.numpy():
    if(k!=result.numpy()[i]):
        plt.figure(figsize=(2,2))
        plt.imshow(test_x.data.numpy()[i].reshape(28,28))
        plt.title(str(result.numpy()[i]))
    i=i+1    
        
accuracy = sum(test_y.data.numpy()==result.numpy())/len(test_y.data.numpy())
print(accuracy)


# In[91]:


#학습시킨 데이터를 cnn_test_model_r1라는 이름으로 저장
savePath = "./data/cnn_test_model_r1.pth"
torch.save(model.state_dict(), savePath)


# In[ ]:




