import torch.nn as nn
import torch.optim as optim
from model import Model 
import random
import tqdm
from utils import preprocess_image, view_image
from IPython.display import clear_output
import os
import matplotlib.pyplot as plt


images_dir = "C:/Mine/ABP/wp"
sample_image = random.sample(os.listdir(images_dir), 1)
num_epochs = 10


image = preprocess_image(os.path.join(images_dir, sample_image[0]))
image = image.float() / 255.0


model = Model()


lr = 1e-3
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)



for epoch in tqdm.tqdm(range(num_epochs), desc="Training Progress"):
  

    print("KKKKKKKKKKKKKKK",image.size())

    output = model(image.unsqueeze(0)) 
    loss = criterion(output, image.unsqueeze(0))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    clear_output(wait=True)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    view_image(image, title=f"Input")
    
    plt.subplot(1, 2, 2)
    view_image(output.squeeze(0).detach(), title=f"Output -> Epoch: {epoch+1}")
    
    plt.tight_layout()
    plt.show()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training completed!")