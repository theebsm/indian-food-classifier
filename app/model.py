import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import io
import os
import gdown

SELECTED_CLASSES = [
    'idli', 'masala_dosa', 'dhokla', 'paani_puri', 'pakode',
    'chai', 'samosa', 'pav_bhaji', 'fried_rice', 'jalebi',
    'butter_naan', 'kadai_paneer', 'kulfi', 'chapati', 'chole_bhature'
]

MODEL_PATH = 'model/efficientnet_indian_food.pth'
FILE_ID = '1Wg3NcdCTOh-xYf6hkP0bSQkqwC2PIvH5'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        os.makedirs('model', exist_ok=True)
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded!")
    else:
        print("✅ Model already exists!")

def load_model():
    download_model()
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 15)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    all_probs = {SELECTED_CLASSES[i]: float(probs[i])
                 for i in range(len(SELECTED_CLASSES))}
    top_idx = probs.argmax().item()
    prediction = SELECTED_CLASSES[top_idx]
    confidence = float(probs[top_idx])
    top3_idx = probs.topk(3).indices.tolist()
    top3 = [{'class': SELECTED_CLASSES[i],
              'confidence': float(probs[i])} for i in top3_idx]
    return {
        'prediction': prediction,
        'confidence': confidence,
        'top3': top3,
        'all_probs': all_probs
    }