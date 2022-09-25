import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import json
import matplotlib.pyplot as plt

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

img = Image.open("tulip.jpg")
plt.imshow(img)
# 图片预处理后会把channel维度放在最前面
img = data_transforms(img)
# 图片变成四维[B,C,H,W]
img = torch.unsqueeze(img,dim=0)

with open("class_indices.json",'r') as f:
    class_indict = json.load(f)


model = AlexNet(num_classes=5)
weight_path = "AlexNet.pth"
model.load_state_dict(torch.load(weight_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output,dim=0)
    predict_cla = torch.argmax(predict).numpy()

print(class_indict[str(predict_cla)],predict[predict_cla].item())
plt.show()




