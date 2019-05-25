#!/usr/bin/env python
# coding: utf-8

# In[44]:


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
import numpy

#책에 나온 mnist말고 다른 사이트(?)를 통해 찾아서 불러옴
mnist = datasets.fetch_mldata('MNIST Original', data_home = './data')

#목적변수를 변수에 할당하고 데이터를 화면에 출력
mnist_label = mnist.target
print(mnist_label)
print(type(mnist_label))

mnist_data = mnist.data/255
#70000행, 784 열 정규화한 data
print(type(mnist_data))

#plt.imshow(mnist_data[0].reshape(28,28),cmap=cm.gray_r)
#plt.show()


# In[45]:


import cv2
img = cv2.imread("./data/811.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
ret, img_th = cv2.threshold(img_blur, 172, 255, cv2.THRESH_BINARY_INV)
#plt.figure(figsize=(12,8))
#plt.imshow(img_th.copy())


contours,_ =cv2.findContours(img_th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
print(type(mnist_label))

rects = [cv2.boundingRect(each) for each in contours ]
tmp = [w*h for (x,y,w,h) in rects]
rects = [(x,y,w,h) for (x,y,w,h) in rects if (w*h>100)]
print(len(rects))

img2 = img.copy()
img_result = []
img_for_class = img.copy()

margin_pixel = 12

for rect in rects:
    #[y:y+h, x:x+w]
    img_result.append(img_for_class[rect[1]-margin_pixel : rect[1]+rect[3]+margin_pixel,
                                    rect[0]-margin_pixel : rect[0]+rect[2]+margin_pixel])
    
    # Draw the rectangles
    cv2.rectangle(img2, (rect[0], rect[1]),
                  (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5) 



# In[46]:


count = 0
nrows = 3
ncols = 4
plt.figure(figsize=(12,10))
for n in img_result:
    count += 1
    if(count%13==0):
        plt.figure(figsize=(12,10))
        count=1
    plt.subplot(nrows, ncols, count)
    test_num = cv2.resize(n, (28,28))[:,:,1]
    #사진 반전 시키고(검은색인 숫자 부분 높은 숫자로) 배경은 수치를 0으로
    test_num = 255- test_num
    test_num = (test_num > 90) * test_num
    test_num = test_num.astype('float32') / 255.
    mnist_data=numpy.vstack((mnist_data,test_num.reshape(1,784)))
    
    plt.imshow(test_num, cmap='Greys', interpolation='nearest');


# In[47]:


print(mnist_data.shape)
print(mnist_label.shape)
label_extend = numpy.array([7,9,8,4,2,5,6,3,1,0,7,9,8,6,3,0,2,5,1,4,9,7,8,4,0,6,2,3,5,1,9,7,8,6,5,4,0,3,2,1,
                            9,7,8,6,3,0,1,2,5,4,9,8,7,0,1,6,3,2,5,4,9,8,7,6,5,4,2,0,3,9,1,8,7,6,4,5,2,0,3,9,
                            1,8,7,6,5,4,2,3,0,1,9,8,7,6,4,5,3,2,0,1,8,9,7,6,5,4,3,2,1,0,7,6,8,9,4,5,3,2,0,1,
                            6,4,5,7,9,3,2,8,0,1,9,7,5,4,6,8,3,2,0,1,9,7,4,8,5,6,3,2,1,0,9,2,8,7,0,4,1,6,3,5,
                            7,9,8,5,6,4,3,0,2,1])
mnist_label=numpy.hstack((mnist_label,label_extend))
print(mnist_label.shape)


# In[61]:


train_size = 10000 #훈련 데이터 건수
test_size = 1000 #테스트 데이터 건수
#데이터 집합을 훈련과 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(mnist_data,mnist_label,train_size=train_size,test_size=test_size)


# In[62]:


train_X = train_X.reshape((len(train_X),1,28,28))
test_X = test_X.reshape((len(test_X),1,28,28))


# In[63]:


#준비가 끝난 데이터를 파이토치가 다룰 수 있는 형태로 정리 tensor생성
#훈련 데이터 텐서 변환
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()
#데스트 데이터 텐서 변환
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()
print(train_X.shape)


# In[64]:


#설명변수와 목적변수 텐서를 합침, 총 train 데이터쌍을 만들기 위해서
train = TensorDataset(train_X, train_Y)
#설명변수 784 개 목적변수 1개
#print(train[0])
#미니배치 분할, 미니배치 학습을 위해 데이터를 셔플링해서 100건 단위로 분할
train_loader = DataLoader(train, batch_size = 100, shuffle =True)


# In[65]:


#신경망 구성, 입력층과 출력층이 1개씩이고, 합성곱층 2개와 풀링층 2개, 전결합층 1개로 구성된 중간층
#입력층의 node 의 수는 28*28, 출력층의 node의 수는 10개
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #합성곱층
        #입력 채널 수, 출력 채널 수, 필터 크기
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        #전결합층
        #256 = (((28-5+1)/2)-5+1)/2*(((28-5+1)/2)-5+1)/2*16
        self.fc1 = nn.Linear(256,64)
        self.fc2 = nn.Linear(64,10)
        
    def forward(self,x):
        #풀링층
        x=F.max_pool2d(F.relu(self.conv1(x)),2)#풀링영역크기
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,256)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x)

model = Net() 
#신경망 인스턴스 생성


# In[66]:


#모형학습
#오차함수 객체
criterion = nn.CrossEntropyLoss()
#최적화를 담당할 객체
optimizer = optim.SGD(model.parameters(),lr=0.01)
#학습시작
for epoch in range(900):
    total_loss = 0
    #분할해 둔 데이터 꺼내오기
    for train_x, train_y in train_loader:
        #계산 그래프 구성
        train_x, train_y = x = variable(train_x),variable(train_y)
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


# In[67]:


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


# In[68]:


print(accuracy)


# In[70]:


print(test_x.shape)
print(test_X.shape)


# In[72]:


#학습시킨 데이터를 cnn_test_model2라는 이름으로 저장
savePath = "./data/cnn_test_model2.pth"
torch.save(model.state_dict(), savePath)


# In[73]:


#앞서 저장한 학습시킨 데이터를 끌어옴
new_model = Net()
new_model.load_state_dict(torch.load(savePath))


# In[ ]:




