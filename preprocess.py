import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

# Cấu hình đường dẫn
DATASET_PATH = './dataset_phishing.csv' 
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
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file {DATASET_PATH}. Hãy tải từ Kaggle và đặt vào thư mục dự án.")
        return

    urls = df['url'].values
    labels = df['status'].values 
    
    # Chuyển về chữ thường
    urls = [s.lower() for s in urls]
    
    # Cấu hình Tokenizer
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    
    # Định nghĩa bộ ký tự (Alphabet)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\_@#$%^&*~+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    tk.word_index = char_dict.copy()
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
    
# Fit tokenizer

    tk.fit_on_texts(urls)
    
    # === QUAN TRỌNG: LƯU TỪ ĐIỂN THUẦN (KHÔNG PHẢI OBJECT KERAS) ===
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        # Chú ý dòng này: tk.word_index
        pickle.dump(tk.word_index, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    print("--> ĐÃ LƯU THÀNH CÔNG FILE: tokenizer.pickle (Phiên bản nhẹ)")

    # Chuyển text sang sequence
    sequences = tk.texts_to_sequences(urls)
    
    # Padding
    maxlen = 37 * 37
    data_pad = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    
    # Chuẩn hóa MinMax
    scaler = MinMaxScaler(feature_range=(0, 255))
    data_scaled = scaler.fit_transform(data_pad)
    data_scaled = np.rint(data_scaled).astype(int)
    
    print("Đang tạo ảnh (Quá trình này có thể mất vài phút)...")
    for i in range(len(data_scaled)):
        # Chuyển thành ma trận 37x37
        matrix_37x37 = data_scaled[i].reshape(37, 37).astype(np.uint8)
        
        # Resize ảnh lên 75x75
        TARGET_SIZE = (75, 75) 
        image = Image.fromarray(matrix_37x37, mode='L') 
        image = image.resize(TARGET_SIZE)
        
        # Lưu ảnh
        label_folder = 'Legitimate' if labels[i] == 'legitimate' else 'Phishing'
        save_path = os.path.join(OUTPUT_DIR, label_folder, f"url_{i}.png")
        image.save(save_path)
        
    print(f"Hoàn tất! Ảnh đã được lưu tại thư mục {OUTPUT_DIR}")

if __name__ == "__main__":
    create_dir_structure()
    preprocess_urls()