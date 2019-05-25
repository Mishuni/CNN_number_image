import torch
from torch.autograd import variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
