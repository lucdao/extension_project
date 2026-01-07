import pandas as pd
import re
from urllib.parse import urlparse
import tldextract
import math
from collections import Counter
from typing import Dict, Any

# --- CẤU HÌNH VÀ THAM SỐ TOÀN CỤC (Giữ nguyên) ---

PHISHING_KEYWORDS = ['login', 'secure', 'update', 'verify', 'account', 'password', 'bank', 'paypal', 'signin']
SHORTENING_SERVICES = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']

# --- 1. HÀM TRÍCH XUẤT ĐẶC TRƯNG TỪNG URL (Giữ nguyên) ---

def extract_all_features(url: str, label: int) -> Dict[str, Any]:
    # ... (Hàm này giữ nguyên như code hoàn chỉnh trước) ...
    # Chỉ sao chép phần logic bên trong hàm để tiết kiệm không gian
    
    # Khai báo các tên đặc trưng để đảm bảo output consistency
    feature_names = [
        'urlLength', 'domainToUrlRatio', 'hasIp', 'hasPort', 'hasHttpWww', 'hasExe', 'hasBackslash', 'maxSub30',
        'dotCount', 'slashRatio', 'specialCharsCount', 'hexCharsCount', 'digitsCount', 'uppercaseCount',
        'vowelsCount', 'consonantsCount', 'hasKeyword', 'hasRedirect', 'hasRef', 'hasAtSymbol', 
        'hasPunycode', 'hasShorteningService', 'domainEntropy', 'homoglyphScore', 'base64Ratio', 'label'
    ]
    
    features = {f: 0 for f in feature_names}

    # 1. Phân tích cấu trúc URL
    try:
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        query = parsed_url.query.lower()
        full_string = url # Giữ nguyên case cho uppercaseCount
        full_string_lower = url.lower()
        
        ext = tldextract.extract(url)
        # Tạo domain_name (không có tiền tố www)
        domain_name = f"{ext.domain}.{ext.suffix}" if ext.suffix else netloc
        
    except Exception:
        # Giữ các đặc trưng bằng 0 nếu URL không hợp lệ
        features['label'] = label
        return features
    
    # Chuỗi chỉ chứa các ký tự chữ và số (dùng cho Vowels/Consonants)
    alpha_string = re.sub(r'[^a-z0-9]', '', full_string_lower)
    total_length = len(full_string_lower)
    domain_length = len(domain_name)
    
    # Ngăn chia cho 0
    if total_length == 0:
        features['label'] = label
        return features
    
    # --- Trích xuất Nhóm I & II (Cấu trúc & Ký tự) ---
    
    # I. CẤU TRÚC CƠ BẢN
    features['urlLength'] = total_length
    features['domainToUrlRatio'] = domain_length / total_length 
    features['hasIp'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', netloc) else 0 
    
    # Sửa lỗi: Kiểm tra sự tồn tại của Port một cách an toàn (tránh lỗi casting)
    try:
        if parsed_url.port is not None:
            features['hasPort'] = 1
        elif ':' in parsed_url.netloc and parsed_url.port is None:
            # Trường hợp có dấu hai chấm nhưng thư viện không trích xuất được port hợp lệ
            features['hasPort'] = 1 
    except ValueError:
        features['hasPort'] = 1
    except TypeError:
        features['hasPort'] = 1
        
    features['hasHttpWww'] = 1 if ('http' in full_string_lower or 'www' in full_string_lower) else 0
    features['hasExe'] = 1 if re.search(r'\.exe$', path) else 0
    features['hasBackslash'] = 1 if '\\' in full_string_lower else 0
    features['maxSub30'] = 1 if max([len(s) for s in re.split(r'[/?&=:#]', full_string_lower) if s], default=0) > 30 else 0

    # II. LEXICAL/SYMBOL
    features['dotCount'] = full_string_lower.count('.')
    features['slashRatio'] = full_string_lower.count('/') / total_length
    
    features['specialCharsCount'] = len(re.findall(r'[^a-z0-9\./:-]', full_string_lower)) 
    features['hexCharsCount'] = len(re.findall(r'%[0-9a-fA-F]{2}', full_string_lower))
    features['digitsCount'] = len(re.findall(r'[0-9]', full_string_lower))
    features['uppercaseCount'] = len(re.findall(r'[A-Z]', full_string)) 
    
    vowels = len(re.findall(r'[aeiou]', alpha_string))
    features['vowelsCount'] = vowels
    features['consonantsCount'] = len(alpha_string) - vowels

    # III. NGỮ CẢNH & TẤN CÔNG
    features['hasKeyword'] = 1 if any(kw in full_string_lower for kw in PHISHING_KEYWORDS) else 0
    features['hasRedirect'] = 1 if any(p in query for p in ['redirect=', 'url=', 'goto=', 'forward=']) else 0
    features['hasRef'] = 1 if any(p in query for p in ['ref=', 'referrer=', 'aff=']) else 0
    features['hasAtSymbol'] = 1 if '@' in full_string_lower else 0

    # IV. CHỐNG GIẢ MẠO (ANTI-SPOOFING)
    
    # 21. hasPunycode (Phát hiện xn--)
    features['hasPunycode'] = 1 if 'xn--' in netloc else 0
    
    # 22. hasShorteningService
    features['hasShorteningService'] = 1 if any(service in netloc for service in SHORTENING_SERVICES) else 0

    # 23. domainEntropy (Entropy Shannon)
    if domain_length > 0:
        domain_counts = Counter(domain_name)
        domain_prob = [float(count) / domain_length for count in domain_counts.values()]
        # Tính Entropy Shannon 
        entropy = -sum([p * math.log2(p) for p in domain_prob if p > 0])
        features['domainEntropy'] = entropy
    
    # 24. homoglyphScore (Đơn giản hóa: đếm ký tự non-ASCII/Unicode)
    features['homoglyphScore'] = len(re.findall(r'[^\x00-\x7F]', url)) 
    
    # 25. base64Ratio (Tỷ lệ chuỗi Base64 có thể có)
    base64_chars = r'[a-zA-Z0-9+/=]'
    base64_candidates = re.findall(f'({base64_chars}{{10,}})', path + query)
    
    max_b64_len = max([len(s) for s in base64_candidates], default=0)
    features['base64Ratio'] = max_b64_len / total_length if total_length else 0

    # Thêm nhãn
    features['label'] = label
    
    return features


# --- 2. HÀM XỬ LÝ CHÍNH (MAIN FUNCTION) ---

def process_data_and_extract_features(input_csv_path: str, output_csv_path: str):
    """
    Đọc dữ liệu từ CSV, trích xuất 25 đặc trưng và lưu vào file CSV mới.
    """
    try:
        # Dùng encoding 'latin-1' để tránh lỗi khi đọc các ký tự đặc biệt (Homoglyph)
        df = pd.read_csv(input_csv_path, encoding='latin-1')
        
        # --- BƯỚC SỬA LỖI: CHUẨN HÓA TIÊU ĐỀ CỘT ---
        
        # 1. Loại bỏ khoảng trắng thừa, ký tự BOM/ẩn, và chuyển sang chữ thường
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False).str.replace('ï»¿', '', regex=False).str.lower()
        
        # 2. Kiểm tra cột URL đã được chuẩn hóa
        if 'url' not in df.columns:
             # Ghi ra các cột hiện tại để hỗ trợ debug
             print("Các cột hiện có trong file CSV (sau chuẩn hóa):", df.columns.tolist())
             raise ValueError("File CSV phải chứa cột 'url' (hoặc 'URL' sau khi chuẩn hóa).")
        
        # Cố gắng xác định cột nhãn
        label_col = None
        for col in ['phishing', 'label', 'status']: # Kiểm tra các tên cột nhãn phổ biến
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
             raise ValueError("File CSV phải chứa cột nhãn (ví dụ: 'phishing', 'label', hoặc 'status').")

        print(f"Đã tải {len(df)} mẫu từ file đầu vào. Cột nhãn: '{label_col}'")
        
        feature_list = []
        
        # Lặp qua từng dòng dữ liệu và trích xuất đặc trưng
        for index, row in df.iterrows():
            url = str(row['url'])
            
            # --- Chuyển đổi nhãn ---
            try:
                label = int(row[label_col]) 
            except ValueError:
                # Xử lý trường hợp nhãn là string (ví dụ: 'legitimate', 'phishing')
                label_str = str(row[label_col]).lower()
                if label_str in ['phishing', 'bad', 'malicious']:
                    label = 1
                else:
                    label = 0
            
            features = extract_all_features(url, label)
            feature_list.append(features)
            
            if (index + 1) % 50000 == 0:
                print(f"Đã xử lý {index + 1} mẫu...")

        features_df = pd.DataFrame(feature_list)
        
        # Lưu DataFrame mới vào file CSV
        features_df.to_csv(output_csv_path, index=False)
        print(f"\nQuá trình trích xuất hoàn tất.")
        print(f"Đã lưu {len(features_df)} mẫu với 25 đặc trưng vào {output_csv_path}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn {input_csv_path}")
    except ValueError as e:
        print(f"Lỗi dữ liệu: {e}")
    except Exception as e:
        print(f"Lỗi không xác định trong quá trình xử lý: {e}")

# --- THỰC THI ---
if __name__ == '__main__':
    # ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN FILE (SỬ DỤNG chuỗi R để tránh lỗi escape ký tự)
    INPUT_FILE_PATH = r"C:\doan2\thuthapdulieu\dulieuchoML\finaldata.csv" 
    OUTPUT_FILE_PATH = r"C:\doan2\thuthapdulieu\dulieuchoML\final2.csv"
    
    print("Bắt đầu trích xuất đặc trưng...")
    process_data_and_extract_features(INPUT_FILE_PATH, OUTPUT_FILE_PATH)