import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import pickle
import io

# === CẤU HÌNH ===
DEVICE = torch.device("cpu")
MODEL_PATH = "resnet18_phishing.pth"
TOKENIZER_PATH = "tokenizer.pickle"
INPUT_SIZE = (75, 75)
MAX_LEN = 37 * 37  # 1369

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# === 1. LOAD TỪ ĐIỂN (DICTIONARY) ===
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        # Load dict thuần thay vì Keras Tokenizer
        word_index = pickle.load(handle) 
    print("Loaded Dictionary successfully!")
except Exception as e:
    print(f"Error loading Tokenizer: {e}")
    word_index = {} # Fallback

# === 2. LOAD MODEL PYTORCH ===
def load_trained_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded Model successfully!")
    except Exception as e:
        print(f"Error loading Model: {e}")
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_trained_model()

# Transform ảnh
pytorch_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === 3. HÀM XỬ LÝ URL THỦ CÔNG (KHÔNG CẦN TENSORFLOW) ===
def manual_url_to_tensor(url):
    # 1. Chuyển chữ thường
    url_lower = url.lower()
    
    # 2. Tokenize (Chuyển ký tự sang số dựa vào word_index)
    seq = []
    for char in url_lower:
        # Lấy index của ký tự, nếu không có thì lấy index của UNK
        idx = word_index.get(char, word_index.get('UNK', 0)) 
        seq.append(idx)
    
    # 3. Padding & Truncating (Cắt hoặc thêm số 0 cho đủ 1369)
    if len(seq) > MAX_LEN:
        seq = seq[:MAX_LEN] # Cắt bớt
    else:
        seq = seq + [0] * (MAX_LEN - len(seq)) # Thêm số 0 vào sau (Padding post)
        
    # 4. Tạo ma trận & Scale màu
    matrix = np.array(seq).reshape(37, 37).astype(np.uint8)
    
    # Scale từ index sang grayscale (MinMax Scaling giả lập)
    # Giả sử vocab size khoảng 40-70 ký tự
    vocab_size = len(word_index) + 1
    matrix_scaled = np.interp(matrix, (0, vocab_size), (0, 255)).astype(np.uint8)
    
    # 5. Tạo ảnh và Tensor
    image = Image.fromarray(matrix_scaled, mode='L')
    image = image.convert("RGB")
    tensor = pytorch_transform(image)
    return tensor.unsqueeze(0).to(DEVICE)

# === API ENDPOINT ===
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

    try:
        data = request.json
        urls = data.get('urls', [])
        response_data = {}
        
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                continue
            try:
                # Dùng hàm xử lý thủ công mới
                tensor_input = manual_url_to_tensor(url)
                
                with torch.no_grad():
                    outputs = model(tensor_input)
                    _, preds = torch.max(outputs, 1)
                    is_phishing = (preds.item() == 1)
                
                if is_phishing:
                    response_data[url] = "PHISHING"
                else:
                    response_data[url] = "SAFE"
            except Exception as e:
                print(f"Error processing {url}: {e}")
                response_data[url] = "ERROR"
                
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)