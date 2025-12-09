import torchvision.transforms as T
import torchxrayvision as xrv
import pydicom
import matplotlib.pyplot as plt

ds = pydicom.dcmread('../../data/box_images/9927054_20051216.dcm')
img = ds.pixel_array

transform = T.Compose([
    T.ToTensor(),                     # → (1,H,W) and scales to [0,1]
    T.Resize((224, 224)),             # resize here
])

img = transform(img)  
img = img.unsqueeze(0)

plt.imshow(img[0][0], cmap = 'gray')
plt.show()

ae = xrv.autoencoders.ResNetAE(weights="101-elastic") # trained on PadChest, NIH, CheXpert, and MIMIC
z = ae.encode(img)
img2 = ae.decode(z)

print(z)

plt.imshow(img2[0][0].detach(), cmap = 'gray')
plt.show()
