import os
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch import nn
import wandb
from PIL import Image
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize

from torch.utils.data import Dataset
def to_long(tensor):
    return tensor.long()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
wandb.login()
wandb.init(project="deeplab-adk20k", name="0506")

# 数据集路径
train_data_path = "A:\\deeplab3_cutimg\\DeepLabV3Plus-Pytorch-master\\data\\train\\img"
train_mask_path = "A:\\deeplab3_cutimg\\DeepLabV3Plus-Pytorch-master\\data\\train\\masks"
val_data_path = "A:\\deeplab3_cutimg\\DeepLabV3Plus-Pytorch-master\\data\\val\\img"
val_mask_path = "A:\\deeplab3_cutimg\\DeepLabV3Plus-Pytorch-master\\data\\val\\masks"

# 超参数设置
num_classes = 211 # 根据您的数据集类别进行设置
batch_size = 8
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = sorted([f for f in os.listdir(img_dir) if not f.endswith(".keep")])
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # 如果未提供图像和掩码转换，使用默认的
        if self.img_transform is None:
            self.img_transform = Compose([
                Resize((256, 256)),  # 调整图像大小
                ToTensor(),
            ])

        if self.mask_transform is None:
            self.mask_transform = Compose([
                Resize((256, 256), Image.NEAREST),  # 调整标签图大小
                ToTensor(),
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_list[idx])
        mask_name = self.img_list[idx].split('.')[0] + '_seg.png' # 更改此行以匹配mask文件名格式
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_name).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask[np.where(mask == 218)] = 1
        mask[np.where(mask == 215)] = 2
        mask[np.where(mask == 214)] = 3
        mask[np.where(mask == 213)] = 4
        mask[np.where(mask == 211)] = 5
        mask = torch.LongTensor(mask)



        if self.img_transform:
            image = self.img_transform(image)

        # if self.mask_transform:
        #     mask = self.mask_transform(mask)

        return image, mask

def remove_dim(x):
    return x.squeeze(0)

# 定义数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((512, 512)), # Add this line
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((512, 512)), # Add this line
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        transforms.Resize((512, 512), Image.NEAREST),
        # transforms.ToTensor(),
        # to_long,  # 将mask转换为long类型
        # transforms.Lambda(remove_dim)  # 使用remove_dim函数
    ])

}




# 加载数据集
train_dataset = CustomDataset(img_dir=train_data_path, mask_dir=train_mask_path, img_transform=data_transforms['train'], mask_transform=data_transforms['mask'])
val_dataset = CustomDataset(img_dir=val_data_path, mask_dir=val_mask_path, img_transform=data_transforms['val'], mask_transform=data_transforms['mask'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 加载模型
model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_classes)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 训练循环
def main():
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            print_every_n_batches = 10
            batch_counter = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                batch_counter += 1
                if batch_counter % print_every_n_batches == 0:
                    print(f"{phase} Batch Loss: {loss.item()}")

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs['out'], 1)
                    loss = criterion(outputs['out'],labels)
                    # print(outputs.keys())


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                print(f"{phase} Batch Loss: {loss.item()}")
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            wandb.log({f"{phase} Loss": epoch_loss, f"{phase} Accuracy": epoch_acc})

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        torch.save(model.state_dict(), str(epoch + 1) + '.pth')
        print()

if __name__ == '__main__':
    main()


    # 保存模型权重
    model_weights_path = "model_weights.pth"
    torch.save(model.state_dict(), model_weights_path)
    wandb.finish()

    print("Training complete.")

