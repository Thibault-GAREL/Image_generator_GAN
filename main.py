import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader,Subset,TensorDataset
from torchvision import datasets, transforms

import sys

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Ou 'Agg' si tu veux un rendu sans interface graphique
# import matplotlib_inline.backend_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
#
# import os
# from google.colab import drive
# from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_use_gnet = "model_parameters_gnet_3.17.pth"
model_use_dnet = "model_parameters_dnet_3.17.pth"

model_save_gnet = "model_parameters_gnet_3.18.pth"
model_save_dnet = "model_parameters_dnet_3.18.pth"

chemin_dataset = 'Dataset_image' #Choisissez ici le chemin du fichier contenant vos images dans votre Google Drive
chemin_images = chemin_dataset + '/image_folder'
chemin_model = chemin_dataset + '/model'
model_path = chemin_model + "/" + model_use_gnet
save_path = chemin_dataset + '/result_image/'


num_epochs = 0 #Choisir un nombre d'épisodes adapté au temps que vous disposez pour éxécuter le programme (30 ==> 1 h)



transform = T.Compose([ T.ToTensor(),
                        T.Resize((64, 64)),
                        T.Normalize([.5,.5,.5],[.5,.5,.5])
                       ])

print(f"chemin : {chemin_images} \n")

dataset = datasets.ImageFolder(root=chemin_images, transform=transform)

#Réduction de la taille du dataset si besoin (Pour libérer de la place dans le CPU)
# n = 3000
# dataset = Subset(dataset,range(n))


batchsize   = 64
data_loader = DataLoader(dataset,batch_size=batchsize,shuffle=True,drop_last=True)

for img, _ in dataset:
    print(img.shape)  # Doit toujours afficher torch.Size([3, 64, 64])
    break


class discriminatorNet(nn.Module):
  def __init__(self):
    super().__init__()

    # convolution layers
    self.conv1 = nn.Conv2d(  3, 64, 4, 2, 1, bias=False)
    self.conv2 = nn.Conv2d( 64,128, 4, 2, 1, bias=False)
    self.conv3 = nn.Conv2d(128,256, 4, 2, 1, bias=False)
    self.conv4 = nn.Conv2d(256,512, 4, 2, 1, bias=False)
    self.conv5 = nn.Conv2d(512,  1, 4, 1, 0, bias=False)

    # batchnorm
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(256)
    self.bn4 = nn.BatchNorm2d(512)

  def forward(self,x):
    x = F.leaky_relu( self.conv1(x) ,.2)
    x = F.leaky_relu( self.conv2(x) ,.2)
    x = self.bn2(x)
    x = F.leaky_relu( self.conv3(x) ,.2)
    x = self.bn3(x)
    x = F.leaky_relu( self.conv4(x) ,.2)
    x = self.bn4(x)
    return torch.sigmoid( self.conv5(x) ).view(-1,1)


dnet = discriminatorNet()
y = dnet(torch.randn(10,3,64,64))
y.shape


class generatorNet(nn.Module):
  def __init__(self):
    super().__init__()

    # convolution layers
    self.conv1 = nn.ConvTranspose2d(100,512, 4, 1, 0, bias=False)
    self.conv2 = nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False)
    self.conv3 = nn.ConvTranspose2d(256,128, 4, 2, 1, bias=False)
    self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
    self.conv5 = nn.ConvTranspose2d(64,   3, 4, 2, 1, bias=False)

    # batchnorm
    self.bn1 = nn.BatchNorm2d(512)
    self.bn2 = nn.BatchNorm2d(256)
    self.bn3 = nn.BatchNorm2d(128)
    self.bn4 = nn.BatchNorm2d( 64)


  def forward(self,x):
    x = F.relu( self.bn1(self.conv1(x)) )
    x = F.relu( self.bn2(self.conv2(x)) )
    x = F.relu( self.bn3(self.conv3(x)) )
    x = F.relu( self.bn4(self.conv4(x)) )
    x = torch.tanh( self.conv5(x) )
    return x


# gnet = generatorNet()
# y = gnet(torch.randn(10,100,1,1))
# print(y.shape)
# pic = y[0,:,:,:].squeeze().detach().numpy().transpose((1,2,0))
# pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))
# plt.imshow(pic)

#Nouveau modèle
# lossfun = nn.BCELoss()

# dnet = discriminatorNet().to(device)
# gnet = generatorNet().to(device)

# d_optimizer = torch.optim.Adam(dnet.parameters(), lr=.002, betas=(.5,.999))
# g_optimizer = torch.optim.Adam(gnet.parameters(), lr=.002, betas=(.5,.999))
# # d_optimizer = torch.optim.Adam(dnet.parameters(), lr=.0002, betas=(.5,.999))
# # g_optimizer = torch.optim.Adam(gnet.parameters(), lr=.0002, betas=(.5,.999))

# len(data_loader)

#Utilisation d'un modèle
import os
os.makedirs(chemin_model, exist_ok=True)
model_path_gnet = os.path.join(chemin_model, model_use_gnet)
model_path_dnet = os.path.join(chemin_model, model_use_dnet)

try:
    dnet = discriminatorNet()
    gnet = generatorNet()
    print("✔️ Modèles DNet et GNet instanciés.")
except Exception as e:
    print(f"❌ ERREUR : Impossible d'instancier les modèles.\n{e}")

# Si Utilisation avec GPU
dnet.load_state_dict(torch.load(model_path_dnet))
gnet.load_state_dict(torch.load(model_path_gnet))
try:
    dnet.load_state_dict(torch.load(model_path_dnet, map_location=device))
    gnet.load_state_dict(torch.load(model_path_gnet, map_location=device))
    print("✔️ Poids des modèles chargés avec succès.")
except Exception as e:
    print(f"❌ ERREUR lors du chargement des poids des modèles.\n{e}")


# Si Utilisation sur le CPU de votre PC
# dnet.load_state_dict(torch.load(model_path_dnet,map_location=torch.device('cpu')))
# gnet.load_state_dict(torch.load(model_path_gnet,map_location=torch.device('cpu')))


dnet = dnet.to(device)
gnet = gnet.to(device)

lossfun = nn.BCELoss()
d_optimizer = torch.optim.Adam(dnet.parameters(), lr=.0002, betas=(.5,.999))
g_optimizer = torch.optim.Adam(gnet.parameters(), lr=.0002, betas=(.5,.999))

len(data_loader)



losses  = []
disDecs = []
training_images = []

dnet.train()
gnet.train()

for epochi in range(num_epochs):

  operations = 0

  for data,_ in data_loader:
    # print(f"Taille du batch reçu : {data.shape[0]}")

    batchsize = data.shape[0]

    operations += 1

    #Création des vecteurs
    data = data.to(device)
    real_labels = torch.ones(batchsize,1).to(device)
    fake_labels = torch.zeros(batchsize,1).to(device)


    #Entrainement du Discriminateur
    pred_real   = dnet(data)
    d_loss_real = lossfun(pred_real,real_labels)

    fake_data   = torch.randn(batchsize,100,1,1).to(device)
    fake_images = gnet(fake_data)
    pred_fake   = dnet(fake_images)
    d_loss_fake = lossfun(pred_fake,fake_labels)

    d_loss = d_loss_real + d_loss_fake

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()



    #Entrainement du Générateur
    fake_images = gnet( torch.randn(batchsize,100,1,1).to(device) )
    pred_fake   = dnet(fake_images)

    g_loss = lossfun(pred_fake,real_labels)

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    losses.append([d_loss.item(),g_loss.item()])

    d1 = torch.mean((pred_real>.5).float()).detach()
    d2 = torch.mean((pred_fake>.5).float()).detach()
    disDecs.append([d1,d2])

    if operations % 20 == 0:
      training_images.append(fake_images[0])

  #Message de suivi de l'entraînement
  msg = f'Épisode {epochi+1}/{num_epochs}'
  sys.stdout.write('\r' + msg)

losses  = np.array(losses)
#disDecs = np.array(disDecs)

if (num_epochs != 0):
    import os
    os.makedirs(chemin_model, exist_ok=True)
    model_path_gnet = os.path.join(chemin_model, model_save_gnet)
    model_path_dnet = os.path.join(chemin_model, model_save_dnet)
    torch.save(gnet.state_dict(), model_path_gnet)
    torch.save(dnet.state_dict(), model_path_dnet)

    print(f"\nModèles sauvegardés :\n - {model_path_gnet}\n - {model_path_dnet}")


#
# def smooth(x,k=15):
#   return np.convolve(x,np.ones(k)/k,mode='same')
#
# fig,ax = plt.subplots(1,3,figsize=(18,5))
#
# ax[0].plot(smooth(losses[:,0]))
# ax[0].plot(smooth(losses[:,1]))
# ax[0].set_xlabel('Batches')
# ax[0].set_ylabel('Loss')
# ax[0].set_title('Model loss')
# ax[0].legend(['Discrimator','Generator'])
#
# ax[1].plot(losses[::5,0],losses[::5,1],'k.',alpha=.1)
# ax[1].set_xlabel('Discriminator loss')
# ax[1].set_ylabel('Generator loss')
#
# # ax[2].plot(smooth(disDecs[:,0]))
# # ax[2].plot(smooth(disDecs[:,1]))
# # ax[2].set_xlabel('Epochs')
# # ax[2].set_ylabel('Probablity ("real")')
# # ax[2].set_title('Discriminator output')
# # ax[2].legend(['Real','Fake'])
#
# plt.show()



def load_generator(model_path, device):
  gnet = generatorNet().to(device)  # Instancie le modèle
  state_dict = torch.load(model_path, map_location=device)  # Charge le state_dict
  gnet.load_state_dict(state_dict)  # Applique les poids au modèle
  gnet.eval()  # Met en mode évaluation
  return gnet

def generate_save_images(gnet, batchsize, device, save_path):
    os.makedirs(save_path, exist_ok=True)

    noise = torch.randn(batchsize, 100, 1, 1).to(device)

    fake_data = gnet(noise).to(device)

    # for i in range(batchsize):
    #     img_tensor = fake_data[i].detach().squeeze().numpy().transpose((1, 2, 0))
    #     img_tensor = (img_tensor - np.min(img_tensor)) / (np.max(img_tensor) - np.min(img_tensor))  # Normalisation
    #     img_array = (img_tensor * 255).astype(np.uint8)  # Convertir en valeurs 0-255
    #
    #     img = Image.fromarray(img_array)  # Convertir en image PIL
    #     img.save(os.path.join(save_path, f"generated_image_{i+1}.png"))  # Sauvegarde
    #
    # print(f"Images sauvegardées dans : {save_path}")

    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    for i, ax in enumerate(axs.flatten()):
        pic = fake_data[i].detach().cpu().squeeze().numpy().transpose((1, 2, 0))
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))  # Normalisation
        ax.imshow(pic, cmap='gray')
        ax.axis('off')

    plt.show()

batchsize = 18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle et génération d'images
gnet = load_generator(model_path, device)
generate_save_images(gnet, batchsize, device, save_path)

gnet.eval()
fake_data = gnet( torch.randn(batchsize,100,1,1).to(device) )

# fig,axs = plt.subplots(3,6,figsize=(12,6))
# for i,ax in enumerate(axs.flatten()):
#   pic = fake_data[i,:,].detach().cpu().squeeze().numpy().transpose((1,2,0))
#   pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))
#   ax.imshow(pic,cmap='gray')
#   ax.axis('off')
#
# plt.show()