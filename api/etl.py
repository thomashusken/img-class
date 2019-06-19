from PIL import Image
import torch
from torchvision import transforms

class ImageParser:
    '''
    Image loader
    '''
    def __init__(self, source, transform=None):
        self.source = source
        self.transform = transform
        self.image = None
        self.processed_image = None

    def load_image(self):
        self.image = Image.open(self.source)
        if self.transform:
            self.processed_image = torch.unsqueeze(self.transform(self.image), 0)
        return self.image, self.processed_image

