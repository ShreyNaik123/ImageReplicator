import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Resize, Lambda
from torchvision.io import read_image
import random
import os
from model import Model
from utils import view_image

model_state_path = 'models\model_20.pth'
model = Model()
model.load_state_dict(torch.load(model_state_path))

images_dir = "C:/Mine/ABP/wp"
num_outputs = 4

random_images = random.sample(os.listdir(images_dir), num_outputs)
print("Random Image", random_images)

plt.figure(figsize=(20, 5 * num_outputs))

for i, random_image in enumerate(random_images):
    image_path = os.path.join(images_dir, random_image)
    image = read_image(image_path)
    resize_transform = Resize((256, 256))
    image = resize_transform(image)
    channel_transform = Lambda(lambda x: x[:3, :, :] if x.size(0) == 4 else x)
    image = channel_transform(image)

    
    # Original image
    plt.subplot(num_outputs, 2, 2*i + 1)
    view_image(image)
    plt.title(f"Original Image {i+1}")
    
    # Model output
    image = image.float() / 255.0
    with torch.no_grad():
        output = model(image.unsqueeze(0)).squeeze(0)
    
    plt.subplot(num_outputs, 2, 2*i + 2)
    view_image(output)
    plt.title(f"Model Output {i+1}")

plt.tight_layout()  
plt.show()




