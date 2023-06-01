import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from pathlib import Path
DeepLabV3Plus = deeplabv3_resnet50
model = DeepLabV3Plus(num_classes=151)
#odel.load_state_dict(torch.load(''))
model.load_state_dict(torch.load("model_weights.pth"), strict=False)

model.eval()


from PIL import Image

image_path = './input/101.jpg'  # 请将此路径替换为您自己的图片路径
image = Image.open(image_path).convert("RGB")
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

# 使用与训练过程相同的图像预处理
img_transform = Compose([
    Resize((1280, 2773)),  # 如果在训练时使用了其他尺寸，请将其更改为相应的尺寸
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_image = img_transform(image).unsqueeze(0)  # 添加批次维度

with torch.no_grad():
    output = model(input_image)
    output_tensor = output['out']
    _, predicted = torch.max(output_tensor, 1)
    segmentation_map = predicted.squeeze(0).cpu().numpy()  # 删除批次维度并将张量移动到CPU


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image)
ax1.set_title('Original Image')

ax2.imshow(segmentation_map, cmap='viridis')  # 您可以根据需要选择不同的色彩映射
ax2.set_title('Segmentation Map')

plt.show()
output_dir = Path("A:\\deeplab3_cutimg\\DeepLabV3Plus-Pytorch-master\\output")
output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
output_path = output_dir / "segmentation_result.png"

# 归一化分割图像的像素值到0-255范围内
segmentation_map_normalized = (segmentation_map - np.min(segmentation_map)) / (np.max(segmentation_map) - np.min(segmentation_map))
segmentation_map_normalized = (segmentation_map_normalized * 255).astype(np.uint8)

# 检查分割图像的最小值和最大值是否相等
# 检查分割图像的最小值和最大值是否相等
if np.min(segmentation_map) != np.max(segmentation_map):
    segmentation_map_normalized = (segmentation_map - np.min(segmentation_map)) / (np.max(segmentation_map) - np.min(segmentation_map))
    segmentation_map_normalized = (segmentation_map_normalized * 255).astype(np.uint8)
else:
    print("The minimum and maximum values in the segmentation map are equal. Skipping normalization.")
    segmentation_map_normalized = segmentation_map.astype(np.uint8)

# 将NumPy数组转换为图像并保存
segmentation_image = Image.fromarray(segmentation_map_normalized)
segmentation_image.save(output_path)

print(f"Segmentation result saved to {output_path}")

