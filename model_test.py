from model import AnomalyAE
from PIL import Image
import torch
from torchvision.transforms import Compose, Grayscale, ToTensor
import matplotlib.pyplot as plt

model = AnomalyAE()
model.load_state_dict(torch.load('./tensorboard_logs_04092023_04-25/models/best_model_25_loss=-0.0000.pt')) # class8

model.eval()
model = model.to('cuda')
# imgpath = "./image/class8/0044.PNG"
imgpath = "./image/class8/0045.PNG"
img = Image.open(imgpath).convert('L')
transform = Compose([Grayscale(), ToTensor()])
img = transform(img)
img = img.to('cuda')
img = img.unsqueeze(0)
y = model(img)
residual = torch.abs(img[0][0]-y[0][0])

plt.figure(figsize=(15,10))
plt.subplot(121)
plt.imshow(img.detach().cpu().numpy()[0][0])
plt.title('Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(residual.detach().cpu().numpy()>0.007)
plt.title('Residual Thresholded')
plt.axis('off')
# plt.savefig('normal.jpg', bbox_inches='tight')
plt.savefig('abnormal.jpg', bbox_inches='tight')


plt.hist(residual.detach().cpu().numpy().ravel())