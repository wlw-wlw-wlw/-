import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

B_SIZE=128
EPOCHS=32
IMG_SIZE=512
TRAIN_ROOT='./data_sets/train'
TEST_ROOT='./data_sets/test'
DEVICE="cpu"
if torch.cuda.is_available:
    DEVICE='cuda'
print(DEVICE)
train_trans = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),  # 调整图像大小为256x256
    #transforms.RandomResizedCrop(244, scale=(0.6, 1.0), ratio=(0.8, 1.0)),  # 随机裁剪图像为244x244
    transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
    transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),  # 改变图像的亮度
    transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),  # 改变图像的对比度
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.4848, 0.4435, 0.4023], std=[0.2744, 0.2688, 0.2757])  # 对图像进行标准化
])
train_set=datasets.ImageFolder(root=TRAIN_ROOT,transform=train_trans)
test_set=datasets.ImageFolder(root=TEST_ROOT,transform=train_trans)
print(train_set.class_to_idx)
train_lder=DataLoader(train_set,batch_size=B_SIZE,shuffle=True)
test_lder=DataLoader(test_set,batch_size=B_SIZE,shuffle=True)

class my_moddel(nn.Module):
    def __init__(self):
        super(my_moddel,self).__init__()
        self.conv1=nn.Conv2d(
            in_channels=3,
            out_channels=9,
            kernel_size=7,
            stride=2,
            padding=3)#9*256*256
        self.conv2=nn.Conv2d(
            in_channels=9,
            out_channels=16,
            kernel_size=5,
            stride=4,
            padding=2)#in: 9*128*128 out:16*32*32
        self.drop1=nn.Dropout(0.1)
        self.drop2=nn.Dropout(0.2)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        #self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(16*32*32,9*32)
        self.fc2=nn.Linear(9*32,12)
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.pool1(x)
        x=self.drop1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=self.drop2(x)
        x=F.relu(x)
        x = torch.flatten(x, 1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.drop1(x)
        x=self.fc2(x)
        output=F.log_softmax(x,dim=1)
        return output

my_test_model=my_moddel().to(DEVICE)
my_test_optimizer=optim.Adam(my_test_model.parameters())#优化器

def train_model(model,device,train_loader,optimizer,epoch):
    model.train()
    for _,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.cross_entropy(output,target)
        #pred=output.max(1,keepdim=1)
        loss.backward()
        optimizer.step()
    print(F'Eporch:{epoch}/{EPOCHS},loss:{loss.item()}')
    # torch.save(model.state_dict(),f'cat_{epoch}.pth')

def test_model(model,device,test_loader):
    model.eval()
    correct,total,test_loss=0,len(test_loader.dataset),0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss=F.cross_entropy(output,target)
            _,pred=torch.max(output,dim=1)
            correct+=pred.eq(target.view_as(pred)).sum().item()
        test_loss=test_loss.item()
        print("Test_averagr loss:{:.4f},Accuracy:{:.3f}\n".format(
            test_loss,correct*100/total))


for data,target in train_lder:
    print(data.shape)
for epoch in range(1,EPOCHS+1):
    train_model(my_test_model,DEVICE,train_lder,my_test_optimizer,epoch)
test_model(my_test_model,DEVICE,test_lder)