# Image generator - GAN

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-2.2.3-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.7.1%2Bcu118-blue.svg)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.10.1-blue.svg)

## 📝 Project Description  
This repository implements a **Generative Adversarial Network (GAN)** using PyTorch.  
It allows training and using a generator model to produce realistic images of **Cat**🐈 from random noise vectors.  
The project is designed to be modular, so you can **train your own GAN** on a custom dataset or **load pre-trained models** to generate images quickly.

For me, I used a dataset of cat pictures to train it ! 😺

---

## ✨ Features / Example Output
- 🎨 **Train a GAN** on your own dataset of images.
- 💾 **Save and reload** both generator and discriminator models.
- 🎲 **Generate batches of fake images** directly from random noise.
- 📊 **Visualization of generated samples** during training.
 

🖼️ Example of generated output :  
![Image of cat](Img/generated_image_5.png) 
![Image of cat](Img/generated_image_10.png) 
![Image of cat](Img/generated_image_14.png)


## ⚙️ **How It Works**

1. **🎨 Generator (GNet)**
   Transforms a **random noise vector** (latent space: 📏 `100`) → **synthetic images** (🖼️ `3x64x64`).

2. **🔍 Discriminator (DNet)**
   Receives **real** or **generated images** → outputs a **probability** of being **"real"** (✅ or ❌).

3. **🔄 Training Loop**
   - **Discriminator**: Learns to **distinguish** real 📸 vs. fake 🤖 images.
   - **Generator**: Learns to **fool** 🎭 the discriminator.
   - **Optimizer**: **Adam** (📈 `lr=0.0002`).

4. **💾 Save & Reuse**
   Models are saved as **`.pth` files** for later **inference** or fine-tuning.

## 📂 Repository Structure  

```bash
/file  
├── Dataset_image/   
│ ├── image_folder/chats &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Folder containing training images / Here, it's cat 😺 !  
│ ├── model/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Saved models (generator & discriminator)  
│ ├── result_image/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Generated results  
├── Img/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Subfolder with the example image for the README.md  
├── LICENSE  
├── README.md  
├── main.py
```

## 💻 Run it on Your PC  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/your-username/gan-image-generator.git
cd gan-image-generator
pip install -r requirements.txt
```

Train the GAN (adjust number of epochs 🔁 in the script):  
```bash
python main.py
```
To **generate** and **display images** using the last models:
```bash
python main.py  #Put num_epochs = 0 l.40
```

## 📖 Inspiration / Sources  
- I follow the learning video : [GAN from NeoCode](https://youtu.be/FWf7NXLjx9c?si=aDglR2UlTNXCPg4-)