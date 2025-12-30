import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ----------------------------
# CNN Model (train.py ile aynı)
# ----------------------------
class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.fc1 = nn.Linear(64*32*32,128)
        self.fc2 = nn.Linear(128,2)
    
    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------
# Model Yükle
# ----------------------------
model = MaskCNN()
model.load_state_dict(torch.load("mask_model.pth", map_location="cpu"))
model.eval()

# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ----------------------------
# Tahmin Fonksiyonu
# ----------------------------
def predict_mask(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return "Maskeli" if pred==0 else "Maskesiz"

# ----------------------------
# Gradio Arayüz
# ----------------------------
interface = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil", label="Yüz Resmi Yükle"),
    outputs="text",
    title="Face Mask Detection",
    description="CNN modeli ile maske takılı mı değil mi tahmin edin."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
