import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os
from sklearn.preprocessing import MinMaxScaler

# Cấu hình đường dẫn
DATASET_PATH = './dataset_phishing.csv' # Thay bằng đường dẫn file csv của bạn
OUTPUT_DIR = 'Images_Data'

def create_dir_structure():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'Legitimate')):
        os.makedirs(os.path.join(OUTPUT_DIR, 'Legitimate'))
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'Phishing')):
        os.makedirs(os.path.join(OUTPUT_DIR, 'Phishing'))

def preprocess_urls():
    print("Đang đọc dữ liệu...")
    # [cite: 797, 806] Đọc dữ liệu
    df = pd.read_csv(DATASET_PATH)
    urls = df['url'].values
    labels = df['status'].values # 'legitimate' hoặc 'phishing'
    
    # [cite: 1349] Chuyển về chữ thường
    urls = [s.lower() for s in urls]
    
    # [cite: 1352] Cấu hình Tokenizer
    # Báo cáo sử dụng char_level=True để xử lý ký tự
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    
    # [cite: 1354] Định nghĩa bộ ký tự (Alphabet)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\_@#$%^&*~+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    tk.word_index = char_dict.copy()
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
    
    # [cite: 1358] Fit tokenizer
    tk.fit_on_texts(urls)
    
    # [cite: 1365] Chuyển text sang sequence
    sequences = tk.texts_to_sequences(urls)
    
    # [cite: 1378] Báo cáo nói ma trận sau xử lý là 37x37 -> độ dài vector là 1369
    maxlen = 37 * 37
    data_pad = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    
    # [cite: 1380] Chuẩn hóa MinMax về [0, 255]
    scaler = MinMaxScaler(feature_range=(0, 255))
    # Flatten để scale rồi reshape lại
    data_scaled = scaler.fit_transform(data_pad)
    data_scaled = np.rint(data_scaled).astype(int)
    
    print("Đang tạo ảnh...")
    for i in range(len(data_scaled)):
        # [cite: 1404] Chuyển thành ma trận 37x37
        matrix_37x37 = data_scaled[i].reshape(37, 37).astype(np.uint8)
        
        # [cite: 1405] Resize ảnh lên 75x75 (Hoặc 100x100 tùy thử nghiệm)
        # Báo cáo code dùng 75x75, nhưng kết luận 100x100 tốt nhất [cite: 1705]
        TARGET_SIZE = (75, 75) 
        image = Image.fromarray(matrix_37x37, mode='L') # Grayscale [cite: 1402]
        image = image.resize(TARGET_SIZE)
        
        # Lưu ảnh vào thư mục tương ứng [cite: 1436]
        label_folder = 'Legitimate' if labels[i] == 'legitimate' else 'Phishing'
        save_path = os.path.join(OUTPUT_DIR, label_folder, f"url_{i}.png")
        image.save(save_path)
        
    print(f"Hoàn tất! Ảnh đã được lưu tại thư mục {OUTPUT_DIR}")

if __name__ == "__main__":
    create_dir_structure()
    preprocess_urls()