from torch import nn

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.autoEncoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
    )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )


  def forward(self, x):
    x = self.autoEncoder(x)
    x = self.decoder(x)
    return x
  


from torch import nn

class Model2(nn.Module):
  def __init__(self):
    super().__init__()
    self.autoEncoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(512, 1024, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(1024, 2048, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
    )

    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
      nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )


  def forward(self, x):
    x = self.autoEncoder(x)
    x = self.decoder(x)
    return x
  
  
  

class Model3(nn.Module):
  def __init__(self):
    super().__init__()
    self.autoEncoder = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
        nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),
        nn.ReLU(True),
 
    )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )


  def forward(self, x):
    x = self.autoEncoder(x)
    x = self.decoder(x)
    return x