import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def classify_image(image_path: str):
    img = Image.open(image_path)
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
    return outputs.argmax(1).item()
