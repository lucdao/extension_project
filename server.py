# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- Bắt buộc dòng này
# ... các import khác (torch, tensorflow...)

app = Flask(__name__)
CORS(app)  # <--- Bắt buộc dòng này để Extension không bị chặn

# ... (Phần code load model và xử lý như cũ) ...

# KHÔNG DÙNG app.run() ở cuối file khi lên Render
# Gunicorn sẽ tự quản lý việc chạy app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)