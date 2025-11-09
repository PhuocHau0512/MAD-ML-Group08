# --- 1. IMPORT CÁC THƯ VIỆN ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Hàm chia data
from sklearn.preprocessing import StandardScaler     # Hàm chuẩn hóa (scaling)
from sklearn.naive_bayes import GaussianNB, MultinomialNB # Hai loại mô hình Naive Bayes
from sklearn.ensemble import RandomForestClassifier # Đây là thư viện triển khai Decision Forest
from sklearn.svm import SVC, LinearSVC              # Mô hình SVM
from sklearn.metrics import classification_report, accuracy_score # Các hàm đo độ chính xác
from sklearn.impute import SimpleImputer             # Hàm xử lý giá trị thiếu (NaN, '?')
import joblib   # Dùng để lưu mô hình
import warnings # Dùng để tắt các cảnh báo
import os       # Dùng để xử lý đường dẫn (os.path.join), tạo thư mục (os.makedirs)
import argparse # Dùng để nhận tham số từ terminal (ví dụ: --output-dir)
import json     # Dùng để ghi tệp cấu hình app_config.json

# Tắt các cảnh báo (warnings) không quan trọng để output được sạch sẽ
warnings.filterwarnings('ignore')

# --- 2. HÀM TRỢ GIÚP: IN BÁO CÁO ĐÁNH GIÁ ---
def print_model_report(model, X_test, y_test, model_name, target_names):
    """
    Hàm trợ giúp để in báo cáo phân loại chi tiết (precision, recall, f1-score)
    cho một mô hình đã được huấn luyện.
    """
    print(f"\n--- Báo cáo chi tiết cho: {model_name} ---")
    try:
        # Dự đoán trên tập dữ liệu kiểm tra (X_test)
        y_pred = model.predict(X_test)
        # Tính độ chính xác tổng thể
        acc = accuracy_score(y_test, y_pred)
        print(f"==> Độ chính xác (Accuracy): {acc * 100:.2f}%")
        # In báo cáo chi tiết
        print(classification_report(y_test, y_pred, target_names=target_names))
    except Exception as e:
        print(f"Lỗi khi tạo báo cáo: {e}")

# --- 3. PHẦN 1: HUẤN LUYỆN DỮ LIỆU PE HEADER ---
def train_pe_header(output_dir):
    """
    Huấn luyện 3 mô hình (NB, DF, SVM) trên dữ liệu PE Header (MalwareData.csv).
    Các đặc trưng này là dạng số thực/số nguyên (continuous/discrete).
    """
    print("="*60)
    print(" BẮT ĐẦU PHẦN 1: HUẤN LUYỆN MÔ HÌNH PE HEADER (MalwareData.csv)")
    print(f"   => Sẽ lưu mô hình vào: {output_dir}")
    print("="*60)
    
    # 3.1. Tải dữ liệu
    try:
        # Tệp này dùng dấu '|' làm dấu phân cách
        df = pd.read_csv("Dataset/MalwareData.csv", sep="|")
    except FileNotFoundError:
        print("[LỖI] Không tìm thấy tệp 'MalwareData.csv'. Bỏ qua Phần 1.")
        return

    # 3.2. Tiền xử lý
    y = df['legitimate'] # Nhãn (0 hoặc 1)
    X = df.drop(['Name', 'md5', 'legitimate'], axis=1) # Đặc trưng
    
    # Lưu lại 54 tên cột đặc trưng. Rất quan trọng cho app.py và feature_extractor.py
    pe_header_columns = list(X.columns)
    joblib.dump(pe_header_columns, os.path.join(output_dir, 'pe_header_columns.joblib'))
    
    # Xử lý giá trị thiếu (NaN) bằng cách thay thế bằng giá trị trung bình (mean)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 3.3. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3.4. Chuẩn hóa (Scaling)
    # Đưa các đặc trưng về cùng thang đo (trung bình = 0, độ lệch chuẩn = 1)
    # Rất quan trọng cho SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Dùng X_test_scaled để đánh giá
    
    # 3.5. Huấn luyện và Lưu
    print("[1/3] Huấn luyện Naive Bayes (Gaussian)...")
    model_nb = GaussianNB() # Dùng GaussianNB vì dữ liệu là số thực liên tục
    model_nb.fit(X_train_scaled, y_train)
    joblib.dump(model_nb, os.path.join(output_dir, 'pe_header_model_nb.joblib'))
    print_model_report(model_nb, X_test_scaled, y_test, "NB (PE Header)", ['Malware (0)', 'Benign (1)'])

    print("[2/3] Huấn luyện Decision Forest (Random Forest)...")
    model_df = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_df.fit(X_train_scaled, y_train)
    joblib.dump(model_df, os.path.join(output_dir, 'pe_header_model_df.joblib'))
    print_model_report(model_df, X_test_scaled, y_test, "DF (PE Header)", ['Malware (0)', 'Benign (1)'])

    print("[3/3] Huấn luyện SVM (LinearSVC) ...")
    # Dùng kernel='linear' và probability=True để có thể giải thích (vẽ biểu đồ)
    model_svm = LinearSVC(random_state=42, dual=False, max_iter=2000)
    model_svm.fit(X_train_scaled, y_train)
    joblib.dump(model_svm, os.path.join(output_dir, 'pe_header_model_svm.joblib'))
    print_model_report(model_svm, X_test_scaled, y_test, "SVM - Linear (PE Header)", ['Malware (0)', 'Benign (1)'])
    
    # Lưu các bộ tiền xử lý (scaler, imputer) để app.py dùng
    joblib.dump(scaler, os.path.join(output_dir, 'pe_header_scaler.joblib'))
    joblib.dump(imputer, os.path.join(output_dir, 'pe_header_imputer.joblib'))
    
    print("[PHẦN 1 HOÀN THÀNH] Đã lưu các tệp .joblib cho PE Header.")

# --- 4. PHẦN 2: HUẤN LUYỆN DỮ LIỆU API IMPORTS ---
def train_api_imports(output_dir):
    """
    Huấn luyện 3 mô hình trên dữ liệu API Imports (1000 đặc trưng).
    Các đặc trưng này là dạng đếm (tần suất).
    """
    print("\n" + "="*60)
    print(" BẮT ĐẦU PHẦN 2: HUẤN LUYỆN MÔ HÌNH API IMPORTS (top_1000_pe_imports.csv)")
    print(f"   => Sẽ lưu mô hình vào: {output_dir}")
    print("="*60)
    
    try:
        df_api = pd.read_csv("Dataset/top_1000_pe_imports.csv")
    except FileNotFoundError:
        print("[LỖI] Không tìm thấy tệp 'top_1000_pe_imports.csv'. Bỏ qua Phần 2.")
        return

    X = df_api.drop(['hash', 'malware'], axis=1)
    y = df_api['malware']
    
    api_imports_columns = list(X.columns)
    joblib.dump(api_imports_columns, os.path.join(output_dir, 'api_imports_columns.joblib'))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[1/3] Huấn luyện Naive Bayes (Multinomial)...")
    model_nb_api = MultinomialNB() # Dùng MultinomialNB vì đây là dữ liệu đếm (count data)
    model_nb_api.fit(X_train, y_train) # NB không cần scale
    joblib.dump(model_nb_api, os.path.join(output_dir, 'api_imports_model_nb.joblib'))
    print_model_report(model_nb_api, X_test, y_test, "NB (API Imports)", ['Benign (0)', 'Malware (1)'])

    print("[2/3] Huấn luyện Decision Forest (Random Forest)...")
    model_df_api = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_df_api.fit(X_train, y_train) # DF/RF cũng không bắt buộc scale
    joblib.dump(model_df_api, os.path.join(output_dir, 'api_imports_model_df.joblib'))
    print_model_report(model_df_api, X_test, y_test, "DF (API Imports)", ['Benign (0)', 'Malware (1)'])

    print("[3/3] Huấn luyện SVM (LinearSVC) ...")
    # SVM *luôn luôn* nên được scale
    scaler_api = StandardScaler()
    X_train_scaled = scaler_api.fit_transform(X_train)
    X_test_scaled = scaler_api.transform(X_test)
    
    model_svm_api = LinearSVC(random_state=42, dual=False, max_iter=2000)
    model_svm_api.fit(X_train_scaled, y_train)
    joblib.dump(model_svm_api, os.path.join(output_dir, 'api_imports_model_svm.joblib'))
    print_model_report(model_svm_api, X_test_scaled, y_test, "SVM - Linear (API Imports)", ['Benign (0)', 'Malware (1)'])

    # Chỉ lưu scaler cho SVM vì 2 mô hình kia không dùng
    joblib.dump(scaler_api, os.path.join(output_dir, 'api_imports_scaler.joblib'))
    
    print("[PHẦN 2 HOÀN THÀNH] Đã lưu các tệp .joblib cho API Imports.")

# --- 5. PHẦN 3: HUẤN LUYỆN DỮ LIỆU DREBIN (APK) ---
def train_apk_drebin(output_dir):
    """
    Huấn luyện 3 mô hình trên bộ dữ liệu Drebin (215 đặc trưng).
    Đã sửa để xử lý giá trị '?' là NaN.
    """
    print("\n" + "="*60)
    print(" BẮT ĐẦU PHẦN 3: HUẤN LUYỆN MÔ HÌNH APK (DREBIN DATASET)")
    print(f"   => Sẽ lưu mô hình vào: {output_dir}")
    print("="*60)
    
    apk_dataset_file = "Dataset/drebin-215-dataset-5560malware-9476-benign.csv"
    
    try:
        # SỬA LỖI: na_values='?' nói cho Pandas biết '?' là giá trị thiếu (NaN)
        df_apk = pd.read_csv(apk_dataset_file, encoding='latin1', low_memory=False, na_values='?')
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy tệp '{apk_dataset_file}'. Bỏ qua Phần 3.")
        return

    try:
        # Map nhãn: S (Malware) -> 0, B (Benign) -> 1
        y = df_apk['class'].map({'S': 0, 'B': 1})
        X = df_apk.drop('class', axis=1)
        
        apk_columns = list(X.columns)
        joblib.dump(apk_columns, os.path.join(output_dir, 'apk_drebin_columns.joblib'))
        
    except KeyError:
        print(f"[LỖI] Không tìm thấy cột 'class'. Bỏ qua Phần 3.")
        return
    
    # SỬA LỖI: Dùng Imputer để thay thế các giá trị '?' (đã thành NaN)
    # Dùng 'median' (trung vị) an toàn hơn 'mean' cho dữ liệu 0/1
    imputer = SimpleImputer(strategy='median') 
    X_imputed = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

    print("[1/3] Huấn luyện Naive Bayes (Multinomial)...")
    model_nb_apk = MultinomialNB()
    model_nb_apk.fit(X_train, y_train)
    joblib.dump(model_nb_apk, os.path.join(output_dir, 'apk_drebin_model_nb.joblib'))
    print_model_report(model_nb_apk, X_test, y_test, "NB (APK Drebin)", ['Malware (S)', 'Benign (B)'])

    print("[2/3] Huấn luyện Decision Forest (Random Forest)...")
    scaler_apk = StandardScaler()
    X_train_scaled = scaler_apk.fit_transform(X_train)
    X_test_scaled = scaler_apk.transform(X_test)
    
    model_df_apk = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_df_apk.fit(X_train_scaled, y_train)
    joblib.dump(model_df_apk, os.path.join(output_dir, 'apk_drebin_model_df.joblib'))
    print_model_report(model_df_apk, X_test_scaled, y_test, "DF (APK Drebin)", ['Malware (S)', 'Benign (B)'])

    print("[3/3] Huấn luyện SVM (LinearSVC) ...")
    model_svm_apk = LinearSVC(random_state=42, dual=False, max_iter=2000)
    model_svm_apk.fit(X_train_scaled, y_train)
    joblib.dump(model_svm_apk, os.path.join(output_dir, 'apk_drebin_model_svm.joblib'))
    print_model_report(model_svm_apk, X_test_scaled, y_test, "SVM - Linear (APK Drebin)", ['Malware (S)', 'Benign (B)'])
    
    # Lưu scaler và imputer
    joblib.dump(scaler_apk, os.path.join(output_dir, 'apk_drebin_scaler.joblib'))
    joblib.dump(imputer, os.path.join(output_dir, 'apk_drebin_imputer.joblib'))
    
    print("\n[PHẦN 3 HOÀN THÀNH] Đã lưu các tệp .joblib cho APK (Drebin).")

# --- 6. HÀM CHÍNH (MAIN) ---
if __name__ == "__main__":
    # Sử dụng argparse để nhận tham số --output-dir từ người dùng
    parser = argparse.ArgumentParser(description="Huấn luyện toàn bộ mô hình cho dự án Malware.")
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='models', # Thư mục mặc định là 'models'
        help="Thư mục để lưu tất cả các tệp .joblib (mô hình, scaler...)"
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    
    # Tạo thư mục nếu nó chưa tồn tại
    if not os.path.exists(output_dir):
        print(f"Thư mục '{output_dir}' không tồn tại. Đang tạo thư mục...")
        os.makedirs(output_dir)
        
    print(f"Bắt đầu quá trình huấn luyện. Tất cả mô hình sẽ được lưu vào: '{output_dir}'")
    
    # Gọi 3 hàm huấn luyện
    train_pe_header(output_dir)
    train_api_imports(output_dir)
    train_apk_drebin(output_dir)
    
    # Ghi đường dẫn thư mục đã chọn vào tệp config
    # để app.py biết chỗ tìm mô hình
    config = {"model_directory": output_dir}
    config_path = 'app_config.json'
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"\nĐã lưu cấu hình đường dẫn ('{output_dir}') vào tệp '{config_path}' cho app.py.")
    except Exception as e:
        print(f"\nLỖI: Không thể ghi tệp cấu hình '{config_path}': {e}")

    print("\n" + "="*60)
    print("    ĐÃ HOÀN THÀNH HUẤN LUYỆN TẤT CẢ CÁC MÔ HÌNH.")
    print(f"   Tất cả các tệp .joblib đã được lưu trong thư mục: '{output_dir}'")
    print("\n   Gõ lệnh sau vào terminal để khởi chạy:")
    print("   streamlit run app.py")
    print("="*60)