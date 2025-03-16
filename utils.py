import matplotlib.pyplot as plt
from torchvision.transforms import transforms, Resize, Lambda
from torchvision.io import read_image
from PIL import Image 

def view_image(tensor, batch=False, title=None):
  if batch:
    tensor = tensor.unsqueeze(0)
  tensor = tensor.permute(1, 2, 0)
  tensor = tensor.detach().numpy()
  if(title):
    plt.title(title)
  plt.imshow(tensor)
  
  


def preprocess_image(image_path):
    image = read_image(image_path)
    resize_transform = Resize((256, 256))
    image = resize_transform(image)
    channel_transform = Lambda(lambda x: x[:3, :, :] if x.size(0) == 4 else x)
    image = channel_transform(image)
    return image


def image_to_tensor(pil_image):
    image_tensor = transforms.ToTensor()(pil_image)
  
    resize_transform = Resize((256, 256))
    image = resize_transform(image_tensor)
    
  
    channel_transform = Lambda(lambda x: x[:3, :, :] if x.size(0) == 4 else x)
    image = channel_transform(image)
    
    return image