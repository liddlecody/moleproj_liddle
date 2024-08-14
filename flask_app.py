from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

model = torch.load('models/final_model.pth', map_location=torch.device('cpu'))
model.eval()