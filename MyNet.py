import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor



class MyNet(nn.Module):
    def __init__(self, in_channel=1, channel1=16, channel2=32, num_classes=10):
        
        super(MyNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channel, channel1, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(channel1, channel2, 5)
        self.pool2 = nn.MaxPool2d(2)
        # input_dim = channel2 * size * size  
        # size = ( ((28-kernal1)+1)/2 - kernal2 + 1  )/2
        fc_dim1 = channel2 * 16
        self.fc1 = nn.Linear(fc_dim1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
        
            
        
        
    def forward(self, x):
        
        temp1 = self.pool1(F.relu(self.conv1(x)))
        temp2 = self.pool2(F.relu(self.conv2(temp1)))
        temp2 = temp2.view(temp2.shape[0], -1)
        scores = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(temp2))))))
        return scores
    
    
    
def train(model,optimizer,num=20, batch_size=512):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = []
    total_acc = []
    total_epoch = []
    for epoch in range(num):
       train_loader = DataLoader(train_dataset, batch_size=batch_size)
       test_loader = DataLoader(test_dataset, batch_size=batch_size)
       for index, (x, y) in enumerate(train_loader):
           
           train_x = x 
           train_y = y
            
           # Forward pass: Compute predicted y by passing x to the model
           y_pred = model(train_x)
           
           # Compute and print loss
           loss = criterion(y_pred, train_y)
           
           #writer.add_scalar("Loss/train", loss, epoch)
           
           # Zero gradients, perform a backward pass, and update the weights.
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           
           
        #    if index % 100 == 0:
        #      print(epoch, index)  
        #      print('Iteration %d, loss = %.4f' % (epoch, loss.item()))
       
       
       total_loss.append(loss.item())
       total_epoch.append(epoch)
       total_test = 0 
       correct_test = 0
    
       for index, (x, y) in enumerate(test_loader):
          test_x = x 
          test_y = y
          predict_y = model(test_x.float()).detach()
          predict_y = torch.argmax(predict_y, -1)
          epoch_correct_num = (predict_y==test_y)
          correct_test += np.sum(epoch_correct_num.numpy(), axis=-1)
          total_test += epoch_correct_num.shape[0]
    
       acc = correct_test/total_test 
       total_acc.append(acc)
       print('accuracy: {:.3f}'.format(acc), flush=True)
       
    return total_epoch, total_acc, total_loss
    
    
# MNIST
# Load the Data
train_dataset = torchvision.datasets.MNIST(root='./train', train=True, transform=ToTensor(), target_transform=None, download=False)
test_dataset = torchvision.datasets.MNIST(root='./train', train=False, transform=ToTensor(), target_transform=None, download=False)

    
if __name__ == '__main__':
    
    writer = SummaryWriter()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    batch_size = 256
    
    
    
    
    
    
    
    
    
    
    
    
    # 设置要进行训练的参数
    learning_rate = [1e-3, 1e-4, 1e-5]
    batch_size = [256, 128, 512]
    num_epoch = [40, 30, 50]
    
    
    # 对训练效果最佳参数和模型进行记录
    best_model = None 
    best_acc = 0 
    total_loss = []
    total_epoch = []
    total_acc = []
    best_lr = 0
    best_batchsize = 0
    best_epoch = 0
    
    # 进行训练并且选取最佳的超参数
    for lr in learning_rate:
        for batch in batch_size:
            for num in num_epoch:
                
                print("learing_rate:{:.5f}, batch_size:{:d}, num_epoch:{:d} " .format( lr, batch, num))
                
                # Create a Module
                model = MyNet()
                print(model)
                
                # We use Adam as optimizer
                #optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
                optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.9, 0.99))
                current_epoch, current_acc, current_loss   = train(model,optimizer, num=num, batch_size=batch)
                current_accurancy = current_acc[-1]
                
                if(current_accurancy > best_acc):
                    best_acc = current_accurancy
                    best_model = model 
                    best_lr, best_batchsize, best_epoch = lr, batch_size, num 
                    total_loss = current_loss
                    total_epoch = current_epoch 
                    total_acc = current_acc
    
                
                
    
    
    
    
    # 显示训练过程中的准确率最高的loss 与 accurancy
    # train_img, train_data = train_dataset._load_data()
    plt.subplot(2,1,1)
    plt.title("Training loss")
    plt.plot(total_epoch, total_loss, color='g', label='training loss')
    plt.xlabel("epoch")
    
    plt.subplot(2,1,2)
    plt.title("Accurancy")
    plt.plot(total_epoch, total_acc, color='b', label='training acc')
    plt.xlabel("acc")
    
    # plt.show()
    
    # 对验证集进行正确度的检测
    
    
    
    # for i,c in enumerate(np.random.randint(0,200,25)):#随机取0，1000里的25张图片
    #     plt.subplot(5,5,i+1)

    #     plt.tight_layout()#调整间距

    #     plt.imshow(train_img[c], interpolation='none')

    #     y_pred = model(train_img)
    #     plt.title("Num Label: {}".format(y_pred[c]))

    #     plt.rcParams['font.sans-serif']=['SimHei']

    # plt.show()

    writer.flush()