#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import cv2

import Image_pick as ip
from AnnClass import Net

#불러올 이미지의 경로와 이름
filePath = "./data/300.jpg"

#학습한 인공지능 불러오기
savePath = "./data/cnn_test_model2.pth"
model = Net()
model.load_state_dict(torch.load(savePath))


# In[29]:


data = []
data = ip.collect_image (filePath, 100, 60, 25000, 140)


# In[30]:



count = 0
nrows = 3
ncols = 4
correct = 0

plt.figure(figsize=(10,8))

for n in data:
    count += 1
    if(count%13==0):
        plt.figure(figsize=(10,8))
        count=1
    plt.subplot(nrows, ncols, count)

    #data로 받은 리스트인 n을 numpy array로 바꿔준다
    test_num = np.array(n)
    test_num = test_num.reshape(28,28)
    plt.imshow(test_num , cmap='Greys', interpolation='nearest');
    
    test_num = test_num.reshape((1, 1, 28, 28))
    test_num = torch.from_numpy(test_num).float()
    #훈련된 인공지능에 이미지를 정보를 넣고
    #결과 값에서 확률이 제일 큰 위치의 index를 뽑아온다
    #index에는 이 이미지가 무슨 숫자로 판단되었는지가 나타난다
    result = torch.max(model(test_num).data,1)[1]
    plt.title(result.item())


# In[ ]:




