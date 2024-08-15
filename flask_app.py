from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


model = models.resnet18()  
num_ftrs = model.fc.in_features


model.fc = nn.Sequential(
    nn.Dropout(0.3),  
    nn.Linear(num_ftrs, 2) 
)


model.load_state_dict(torch.load('models/final_model.pth', map_location=torch.device('cpu')))
model.eval()


class_names = ['benign', 'malignant']  

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(file.stream)
    img = transform(img).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(img)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        prediction = 'benign' if predicted_class.item() == 0 else 'malignant'
        confidence_score = confidence.item()

    return jsonify({'prediction': prediction, 'confidence': confidence_score})

if __name__ == '__main__':
    app.run(debug=True)
