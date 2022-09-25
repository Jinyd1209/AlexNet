import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms,datasets,utils
from torch.utils.data import DataLoader
import torch.optim as optim
from model import AlexNet
import os
import json
import time
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # normalize to [0,1]
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),  # three channels normalize to [-1,1]

    "val":transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
}

data_root = os.getcwd()
image_path = data_root + "\\flower_data"
train_dataset = datasets.ImageFolder(root=image_path+"\\train",
                                     transform=data_transform['train'])
val_dataset = datasets.ImageFolder(root=image_path+"\\val",
                                   transform=data_transform['val'])
train_num = len(train_dataset)
val_num = len(val_dataset)
flower_list = train_dataset.class_to_idx  # 将每一个类别对应一个索引值，返回一个字典
cla_dict = dict((val,key) for key,val in flower_list.items())
# 将字典编码成json格式
json_str = json.dumps(cla_dict,indent=4) # 把字典转换成str数据类型后再写入
with open('class_indices.json','w') as json_file:
    json_file.write(json_str)
# 等价写法
with open('class_indices.json','w') as f:
    json.dump(cla_dict,f)

batch_size = 32
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                          num_workers=0)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,
                        num_workers=0)

# test_img,test_label = next(iter(val_loader))
#
#
# def img_show(img):
#     img = img /2 +0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()
#
#
# print(' '.join('%5s'%cla_dict[test_label[j].item()] for j in range(4)))
# img_show(utils.make_grid(test_img))  # make_grid将几张图片合为一张图片

net = AlexNet(num_classes=5,init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.0002)

save_path = '.\\AlexNet.pth'
best_acc = 0.0
for epoch in range(40):
    net.train()
    total_loss = 0.0
    t1 = time.perf_counter()
    for batch,data in enumerate(train_loader):
        images,labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = net(images)
        loss = loss_function(preds,labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # 可视化训练进度
        rate = (batch+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "-" * int((1-rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss),end="")

    print()
    print(time.perf_counter()-t1)

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for _,val_data in enumerate(val_loader):
            val_images,val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            outputs = net(val_images)
            pred_labels = torch.argmax(outputs,dim=1)
            acc+=(pred_labels == val_labels).sum().item()

        average_acc = acc / val_num
        if average_acc > best_acc:
            best_acc = average_acc
            torch.save(net.state_dict(),save_path)

        print("[epoch %d] train_loss: %.3f test_acc: %.3f" %(epoch+1,total_loss/len(train_loader),average_acc))

print("finished training!")





