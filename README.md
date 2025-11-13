# PHÂN TÍCH VÀ PHÁT HIỆN PHẦN MỀM ĐỘC HẠI BẰNG MACHINE LEARNING 

Dự án nghiên cứu và xây dựng hệ thống phát hiện phần mềm độc hại (malware) sử dụng các thuật toán Machine Learning. 
Hệ thống hỗ trợ phân tích đặc trưng tĩnh của tệp PE (Windows), cho phép người dùng **tải lên tệp `.exe` để trích xuất đặc trưng tự động** hoặc tải lên tệp CSV. 
Đối với Android, hệ thống phân tích **bộ dữ liệu Drebin** (từ CSV).

------------------------------------------------------------------------------------------------------------------------------------------

## MỤC LỤC

1.  Tính năng chính
2.  Cấu trúc dự án
3.  Yêu cầu hệ thống
4.  Cài đặt và chạy
5.  Cách sử dụng ứng dụng
6.  Kết quả mô hình (Tóm tắt)
7.  Hướng phát triển tiếp
8.  Gợi ý debug / lưu ý kỹ thuật
9.  Thông tin nhóm

------------------------------------------------------------------------------------------------------------------------------------------

## 1. TÍNH NĂNG CHÍNH

-   **Phân tích đa dạng:**
    1.  PE Header (từ `.exe` hoặc CSV).
    2.  PE API Imports (từ `.exe` hoặc CSV).
    3.  APK Features (Drebin CSV).

-   **Thuật toán:** 
    1. Naive Bayes (Gaussian và Multinomial).
    2. SVM (Support Vector Machine) với LinearSVC để dễ giải thích.
    3. Decision Forest (RandomForestClassifier).

-   **Trích xuất đặc trưng tự động:** Tích hợp `pefile` để tự động đọc và trích xuất đặc trưng PE Header và API Imports trực tiếp từ tệp `.exe`

-   **Trực quan hóa:** 
    1. Biểu đồ tròn tóm tắt tỉ lệ Malware/Lành tính.
    2. Biểu đồ 20 đặc trưng quan trọng nhất cho từng mô hình

-   **Quản lý mô hình:** cho phép chỉ định `--output-dir` để lưu các tệp mô hình `.joblib` khi huấn luyện                                   

------------------------------------------------------------------------------------------------------------------------------------------

## 3. YÊU CẦU HỆ THỐNG

Python 3.8+, thư viện liệt kê trong `requirements.txt`.

**Hướng dẫn bao gồm tạo venv, cài thư viện, huấn luyện mô hình, chạy ứng dụng web bằng Streamlit**

**Tạo venv**
Cách 1: 
```bash
python -m venv venv
```

Cách 2: Ctrl + Shift + P

**Kích hoạt (Windows PowerShell)**
Cách 1:
```bash 
.\.venv\Scripts\Activate
```

Cách 2: 
```bash
.\.venv\Scripts\Activate.ps1
```

**Hoặc Linux / MacOS**
```bash
source venv/bin/activate
```
------------------------------------------------------------------------------------------------------------------------------------------

## 4. CÀI ĐẶT VÀ CHẠY

**Bước 1 — Chuẩn bị thư mục**

1. Tạo một thư mục dự án.

2. Sao chép các tệp code sau vào thư mục: requirements.txt, train_all_models.py, feature_extractor.py, app.py

3. Sao chép các tệp dữ liệu CSV chính vào cùng thư mục:

 - MalwareData.csv

 - top_1000_pe_imports.csv

 - drebin-215-dataset-5560malware-9476-benign.csv

 **Bước 2 — Cài đặt thư viện (chỉ làm 1 lần)**
```bash
 pip install -r requirements.txt
```

**Bước 3 — Huấn luyện mô hình (chỉ làm 1 lần)**
1. Lựa chọn A (mặc định, đơn giản): dùng thư mục models mặc định
```bash
python train_all_models.py
```

2. Lựa chọn B (nâng cao): chỉ định thư mục lưu mô hình
```bash
python train_all_models.py --output-dir my_model_files
```

**Bước 4 — Khởi chạy ứng dụng web**
```bash
streamlit run app.py
```

**Mở trình duyệt theo URL hiển thị trong terminal (mặc định http://localhost:8501).**
------------------------------------------------------------------------------------------------------------------------------------------

## 5. CÁCH SỬ DỤNG

Ứng dụng hỗ trợ phân tích PE Header, API Imports và APK Drebin qua `.exe` hoặc `.csv`.

**Sử dụng menu bên trái (sidebar) để chọn 1 trong 3 loại phân tích (PE Header, PE API Imports, APK/Drebin).**

**Trong thẻ (card) chính, chọn thuật toán muốn dùng: Decision Forest (DF), SVM-Linear, Naive Bayes (NB).**

**Chọn Tab phương thức tải lên:**
1. ```PE Header / API Imports```:
    - Tab 1: 📁 Tải lên tệp .exe (Tự động) — hệ thống sẽ trích xuất đặc trưng và dự đoán.

    - Tab 2: 📄 Tải lên tệp CSV (Thủ công) — tải file CSV có định dạng giống file huấn luyện.

2. ``APK (Android)``: Chỉ hỗ trợ 📄 Tải lên CSV (định dạng giống drebin-215-dataset-...csv).

**Xem kết quả dự đoán, biểu đồ tỉ lệ Malware/Lành tính và biểu đồ 20 đặc trưng quan trọng ngay bên dưới.**
------------------------------------------------------------------------------------------------------------------------------------------

## 6. KẾT QUẢ MÔ HÌNH
Kết quả lấy từ train_all_models.py 

**Phân tích PE Header (MalwareData.csv)**

-   Decision Forest (Random Forest): ~99.54%

-   SVM (Linear): ~97.92%

-   Naive Bayes (Gaussian): ~46.51%

**Phân tích API Imports (top_1000_pe_imports.csv)**

-   Decision Forest: ~98.21%

-   SVM (Linear): ~98.69%

-   Naive Bayes (Multinomial): ~85.25%

**Phân tích APK (Drebin)**

-   Decision Forest (đã scale): ~98.60%

-   SVM (Linear): ~98.40%

-   Naive Bayes (Multinomial): ~97.41%

------------------------------------------------------------------------------------------------------------------------------------------

## 7. HƯỚNG PHÁT TRIỂN

-   Tự động trích xuất 215 đặc trưng Drebin từ file .apk bằng androguard.

-   Phân tích động: xây dựng sandbox (ví dụ: Cuckoo) để thu thập hành vi runtime.

-   Deep Learning: áp dụng CNN trên ảnh biểu diễn nhị phân của file hoặc LSTM cho chuỗi opcode.

------------------------------------------------------------------------------------------------------------------------------------------

## 8. GỢI Ý DEBUG / LƯU Ý KỸ THUẬT
-   feature_extractor.py đã được cập nhật để xử lý vấn đề file lock khi đọc .exe.

-   train_all_models.py đã sửa lỗi liên quan đến DataFrame và SVM-Linear chạy ổn định.

-   Nếu thiếu matplotlib, chạy lại ```pip install -r requirements.txt``` hoặc cài riêng: 
```bash
pip install matplotlib
```
------------------------------------------------------------------------------------------------------------------------------------------
## 9. THÔNG TIN NHÓM

-   Nguyễn Thị Mỹ Duyên - 2033220774
-   Lê Phước Hậu - 2033221314

## License

Tài liệu này và mã nguồn dự án được chia sẻ cho mục đích học tập. Vui lòng tham khảo bản quyền riêng (nếu cần) trước khi sử dụng cho mục đích thương mại.

## Liên hệ

Nếu cần hỗ trợ thêm hoặc muốn chỉnh sửa README, liên hệ:

-   Email: ph124work@gmail.com hoặc giamy26052004@gmail.com

-   Hoặc mở issue / pull request trên repository.
