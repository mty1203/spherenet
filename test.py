import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import time
import os
import psutil
from torchsummary import  summary
LEARNING_RATE_Max = 0.0004
#import argparse
#from spherenet import OmniMNIST, OmniFashionMNIST
from torch import nn
import torch.nn.functional as F
from SphereConv2d import SphereConv2d 
from SphereMaxPool2d import SphereMaxPool2d
NUM_EPOCHS =40
#loading data .npy file (B,H,W) equitangular
theta= np.load("../temp/y_p.npy")[:,0]
#theta=theta[:,np.newaxis]
y_all = theta#np.concatenate((np.sin(theta),np.cos(theta)),axis=1)
#energy=np.load("/home/duyang/meng/txt/y_7.npy")[:,6]
x_slope=np.load("../temp/slope_p.npy")[:,:,:]
x_npe=np.load("../temp/npe_p.npy")[:,:,:]
x_fht=np.load("../temp/fht_p.npy")[:,:,:]
x_peak = np.load("../temp/peak_p.npy")[:,:,:]
#x_nperatio = np.load("../temp/nperatio_p.npy")[:,:,:]
x_slope=x_slope[:,np.newaxis,:,:]
x_npe=x_npe[:,np.newaxis,:,:]
x_fht=x_fht[:,np.newaxis,:,:]
x_peak = x_peak[:,np.newaxis,:,:]
#x_nperatio = x_nperatio[:,np.newaxis,:,:]

x_all=np.concatenate((x_slope,x_npe,x_fht,x_peak),axis=1)
del x_fht
del x_npe
del x_slope
del x_peak
#del x_nperatio
x_all[x_all==1250]=0

print("y_shape={},x_shape={}".format(y_all.shape,x_all.shape))

train_data = []
for i in range(len(y_all)):
   train_data.append([x_all[i,:,:,:], y_all[i]])

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
#i1, l1 = next(iter(trainloader))
#print(i1.shape)

train_size = int(len(train_data)*0.8)  
test_size = len(train_data) - train_size     

train_dataset, test_dataset =random_split(train_data, [train_size, test_size])

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 32, 'pin_memory': True} if use_cuda else {}
#preprocessing

batch_size=16
test_batch_size=16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

print("Length of the train_loader:", len(train_loader))
print("Length of the val_loader:", len(test_loader))
#print(train_data[0].size()) 

images, labels = next(iter(test_loader))
print("images.shape: ",images.shape)
print("labels.shape: ",labels.shape)

#load data complete

class SphereNet(nn.Module):
    def __init__(self):
        super(SphereNet, self).__init__()
        self.conv1 = SphereConv2d(4, 32,kernel_size=(3, 3),stride=1) #use four chanels FHT SLOPE NPE NPERatio
        self.pool1 = SphereMaxPool2d(stride=2)
        self.conv2 = SphereConv2d(32, 64, kernel_size=(3, 3),stride=1)
        self.pool2 = SphereMaxPool2d(stride=2)
        self.conv3 = SphereConv2d(64,128, kernel_size=(3, 3),stride=1)
        self.conv4 = SphereConv2d(128,256, kernel_size=(3, 3),stride=1)
        self.conv5 = SphereConv2d(256,256, kernel_size=(3, 3),stride=2)
        self.conv8 = SphereConv2d(256,512, kernel_size=(3, 3),stride=1)
        self.conv6 = SphereConv2d(256,256, kernel_size=(3, 3),stride=1)
        self.conv7 = SphereConv2d(128,256, kernel_size=(3, 3),stride=2)
        self.dropout3=nn.Dropout(0.25)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.batch_norm4 = nn.BatchNorm2d(1024)
        self.conv2d=nn.Conv2d(512, 1024,3 )
        self.fc1 = nn.Linear(1024, 256)
        self.batchnorm4 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,1)
        self.conv2d1 = nn.Conv2d(512,512,kernel_size=(1,8))
        self.m =nn.AvgPool2d((6,7))
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(self.conv2(x))
        x= F.relu(self.batch_norm1(x))   #first block 63*126*64
        x = self.conv3(x)
        x = self.batch_norm2(self.conv7(x))
        x= F.relu(x) #second block 32*63*256
        x = F.relu(self.pool2(self.batch_norm2(self.conv6(x))))
        x = F.relu(self.pool2(self.batch_norm2(self.conv6(x))))  #16*32*256
        x=  F.relu(self.batch_norm3(self.conv8(x)))  #8*16*512
        x=  F.relu(self.batch_norm3(self.conv2d1(x)))  #8*9*512
        x = F.relu(self.batch_norm4(self.conv2d(x))) #6*7*512
       #x = self.dropout3(x)#drop 0.25
        #x =self.m(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.view(-1, 1024)
       # x = x.view(-1, 10752)  # flatten, [B, C, H, W) -> (B, C*H*W)
       # x = self.dropout3(x)
        x = F.relu(self.fc1(x))
        x =F.relu(self.fc2(x))   #a angle
        x=self.fc3(x)
        return x
#model define finished



#keep track of losses

def train(model, device, train_loader, optimizer, epoch,lr_sched):
    model.train()
    losses = []
    epoch_loss = 0
    lrs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data= data.float()
        output = model(data)
        #loss = F.cross_entropy(output, target)
        loss = criterion(torch.flatten(output), torch.flatten(target))
        losses.append(loss.item())
        epoch_loss =(np.mean(losses)).tolist()  
#backward and optimize
        optimizer.zero_grad()
        loss.backward()
#可以加入梯度裁剪  以后
        nn.utils.clip_grad_value_(model.parameters(), 0.1) 
        optimizer.step()
        if batch_idx % 100== 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        lrs.append( optimizer.param_groups[0]["lr"] )
        lr_sched.step()  
    return epoch_loss ,  lrs

def test(model, device, test_loader):
    model.eval()
    test_loss = []
    epoch_loss = 0
    v_pre = []
    v_tru = []
    v_pre2 = []
    v_tru2 = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #if data.dim() == 3:
             #   data = data.unsqueeze(1)  # (B, H, W) -> (B, C, H, W)
            output = model(data)
            loss = criterion(torch.flatten(output), torch.flatten(target))
            test_loss.append(loss.item())
            epoch_loss =(np.mean(test_loss)).tolist()
            #test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            for idx in range(output.size(0)):
                v_pre.append(output[idx][0].detach().cpu().numpy())
                v_tru.append(target[idx].detach().cpu().numpy())
                #v_pre2.append(output[idx][1].detach().cpu().numpy())
                #v_tru2.append(labels[idx][1].detach().cpu().numpy())
            #correct += pred.eq(target.view_as(pred)).sum().item()

    #test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f})\n'.format(
        epoch_loss,))
    return epoch_loss, v_tru, v_pre#,v_tru2,v_pre2
   

model = SphereNet()
#if torch.cuda.device_count() > 1:
 #   print("Use", torch.cuda.device_count(), "GPUs")
  #  model = nn.DataParallel(model)   
model.to(device)
summary(model,input_size=(4,126,252))

criterion = nn.SmoothL1Loss(beta=1)
##criterion = nn.HuberLoss(delta=0.1) pytorch version >=1.9.0
#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE_Max, weight_decay=WEIGHT_DECAY)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_Max, weight_decay=2e-5 )
# one-cycle learning rate scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE_Max, final_div_factor=1e2, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))


#start training 
start =time.time()
print (f'memory usage： {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024 :.4f} GB' ) 

# record training process
epoch_train_losses = []
epoch_test_losses = []
epoch_test_tru = []
epoch_test_pre = []
epoch_test_tru2 = []
epoch_test_pre2 = []
epoch_best = -1
epoch_best_loss = 1e0

## one-cycle learning rate scheduler
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE_Max, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader))

epoch_Learning_Rates = []
    
# start training
for epoch in range(NUM_EPOCHS):
    # train, test model
    train_losses, Learning_rate = train(model, device, train_loader, optimizer, epoch, scheduler)
    #test_losses, test_tru, test_pre, test_tru2, test_pre2  = validation(model, device, optimizer, test_loader)
    test_losses, test_tru, test_pre = test(model, device, test_loader)#,test_tru2 ,test_pre2
    
    #scheduler.step()        

    if(test_losses<epoch_best_loss) :
        epoch_best_loss=test_losses
        epoch_best=epoch
    
    # save results
    epoch_Learning_Rates.append(Learning_rate)

    epoch_train_losses.append(train_losses)
    epoch_test_losses.append(test_losses)
    epoch_test_tru.append(test_tru)
    epoch_test_pre.append(test_pre)
    #epoch_test_tru2.append(test_tru2)
    #epoch_test_pre2.append(test_pre2)

     # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_test_losses)
    C = np.array(epoch_test_tru)
    D = np.array(epoch_test_pre)
    #G = np.array(epoch_test_tru2)
    #H = np.array(epoch_test_pre2)
    E = np.array(epoch_Learning_Rates)
np.save("train_loss.npy",A)
np.save("test_loss.npy",B)
np.save("test_true.npy",C)
np.save("pre.npy",D)
#np.save("test_true2.npy",G)
#np.save("pre2.npy",H)
np.save("learning_rate",E)
end = time.time()
print('\n Running time: %s min'%((end-start)/60))    
print(f'epoch_best_loss = {epoch_best_loss:.6f}, epoch_best = {epoch_best:.0f}, lr_best = {epoch_Learning_Rates[epoch_best][-1]:.2E}')    
print("done!")


