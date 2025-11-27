import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import pickle
import io

# === CẤU HÌNH ===
DEVICE = torch.device("cpu") # Render Free dùng CPU
MODEL_PATH = "resnet18_phishing.pth"
TOKENIZER_PATH = "tokenizer.pickle"
INPUT_SIZE = (75, 75)

app = Flask(__name__)
# Cấu hình CORS chấp nhận mọi request
CORS(app, resources={r"/*": {"origins": "*"}})

# === LOAD TÀI NGUYÊN ===
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Loaded Tokenizer")
except:
    print("Error loading Tokenizer")

def load_trained_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded Model")
    except:
        print("Error loading Model")
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_trained_model()

pytorch_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def url_to_tensor(url):
    url_lower = url.lower()
    seq = tokenizer.texts_to_sequences([url_lower])
    maxlen = 37 * 37
    data_pad = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')
    matrix = data_pad[0].reshape(37, 37).astype(np.uint8)
    vocab_size = len(tokenizer.word_index) + 1
    matrix_scaled = np.interp(matrix, (0, vocab_size), (0, 255)).astype(np.uint8)
    image = Image.fromarray(matrix_scaled, mode='L')
    image = image.convert("RGB")
    tensor = pytorch_transform(image)
    return tensor.unsqueeze(0).to(DEVICE)

# === API ENDPOINT (ĐÃ SỬA) ===
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Xử lý CORS Preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

    # Xử lý Logic chính
    try:
        data = request.json
        urls = data.get('urls', [])
        response_data = {}
        
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                continue
            try:
                tensor_input = url_to_tensor(url)
                with torch.no_grad():
                    outputs = model(tensor_input)
                    _, preds = torch.max(outputs, 1)
                    # 1 là Phishing (theo logic train)
                    is_phishing = (preds.item() == 1)
                
                if is_phishing:
                    response_data[url] = "PHISHING"
                else:
                    response_data[url] = "SAFE"
            except Exception as e:
                print(f"Error: {e}")
                response_data[url] = "ERROR"
                
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)