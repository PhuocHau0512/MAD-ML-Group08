# --- 1. IMPORT CÁC THƯ VIỆN ---
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import tempfile
import feature_extractor # Import tệp code feature_extractor.py của chúng ta
import json              # Import json để đọc tệp app_config.json
import matplotlib.pyplot as plt # Import để vẽ biểu đồ

# --- 2. CẤU HÌNH TRANG & CSS TÙY CHỈNH ---
st.set_page_config(
    page_title="Phân tích Malware", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🛡️" # Thêm icon cho tab trình duyệt
)

def load_css():
    """Tải CSS tùy chỉnh để làm đẹp giao diện"""
    css = """
    /*--- Nền chính (màu xám nhạt) ---*/
    [data-testid="stAppViewContainer"] > .main {
        background-color: #F0F2F6;
    }
    /*--- Sidebar (màu xanh đậm) ---*/
    [data-testid="stSidebar"] {
        background-color: #0D1B2A;
        border-right: 2px solid #E0E0E0;
    }
    /*--- Chữ trên Sidebar (màu trắng) ---*/
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .st-eb {
        color: #FAFAFA;
    }
    /*--- Thẻ Info trên Sidebar ---*/
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #4A6D7C;
        border-radius: 8px;
    }
    /*--- Tiêu đề chính (màu xanh) ---*/
    h1 { color: #1E3A8A; font-weight: bold; }
    h2 { color: #1E3A8A; }
    h3 { color: #3182CE; }
    /*--- Thẻ (Card) nội dung (màu trắng, bo góc, đổ bóng) ---*/
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 25px 25px 35px 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap: 0rem; }
    /*--- Tabs (chọn file .exe / .csv) ---*/
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background-color: #F0F2F6;
        color: #555555;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        font-weight: bold;
        color: #3182CE;
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Chạy hàm tải CSS
load_css()


# --- 3. TẢI CẤU HÌNH & MÔ HÌNH ---
CONFIG_FILE = 'app_config.json'
DEFAULT_MODEL_DIR = 'models' 

@st.cache_data # Cache: Giúp Streamlit không cần tải lại mô hình mỗi khi có tương tác
def get_model_directory():
    """
    Đọc tệp app_config.json để tìm đường dẫn thư mục mô hình.
    Nếu không thấy, dùng thư mục 'models' làm mặc định.
    """
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_MODEL_DIR
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        model_dir = config.get('model_directory', DEFAULT_MODEL_DIR)
        return model_dir
    except Exception:
        return DEFAULT_MODEL_DIR

# Lấy đường dẫn thư mục mô hình
MODEL_DIRECTORY = get_model_directory()


@st.cache_data
def load_assets(model_type, model_dir):
    """
    Tải 3 mô hình (DF, SVM, NB) và các tệp (scaler, imputer, columns)
    cho loại phân tích được chọn (ví dụ: 'pe_header') từ thư mục 'model_dir'.
    """
    assets = {} # Tạo một dictionary để chứa tài sản
    prefix = model_type
    
    try:
        # Tải 3 mô hình
        assets['model_df'] = joblib.load(os.path.join(model_dir, f'{prefix}_model_df.joblib'))
        assets['model_svm'] = joblib.load(os.path.join(model_dir, f'{prefix}_model_svm.joblib'))
        assets['model_nb'] = joblib.load(os.path.join(model_dir, f'{prefix}_model_nb.joblib'))
        # Tải danh sách cột
        assets['columns'] = joblib.load(os.path.join(model_dir, f'{prefix}_columns.joblib'))
    except FileNotFoundError:
        st.error(f"LỖI: Không tìm thấy tệp mô hình cho '{prefix}' trong thư mục '{model_dir}'.")
        st.warning("Vui lòng chạy lại `python train_all_models.py`")
        return None

    # Tải các tệp tiền xử lý (nếu có)
    scaler_path = os.path.join(model_dir, f'{prefix}_scaler.joblib')
    imputer_path = os.path.join(model_dir, f'{prefix}_imputer.joblib')
    
    if os.path.exists(scaler_path):
        assets['scaler'] = joblib.load(scaler_path)
    if os.path.exists(imputer_path):
        assets['imputer'] = joblib.load(imputer_path)
        
    return assets

# --- 4. HÀM VẼ BIỂU ĐỒ ---
def plot_results(df_display, assets, model_name):
    """
    Vẽ 2 biểu đồ: Biểu đồ tròn (Tóm tắt) và Biểu đồ cột (Độ quan trọng).
    """
    st.markdown("<h3>📊 Trực quan hóa Kết quả</h3>", unsafe_allow_html=True)

    # Chia layout thành 2 cột: 1 cho biểu đồ tròn, 2 cho biểu đồ cột
    col1, col2 = st.columns([1, 2])

    # --- Biểu đồ 1: Tóm tắt kết quả (Biểu đồ tròn) ---
    with col1:
        st.markdown("#### Tóm tắt Dự đoán")
        # Đếm số lượng 'Malware' và 'An toàn'
        result_counts = df_display['Kết quả'].value_counts()
        
        labels = result_counts.index
        sizes = result_counts.values
        # Gán màu: Đỏ cho Malware, Xanh cho An toàn
        colors = ['#FF4B4B' if 'Malware' in label else '#00C49A' for label in labels]
        
        fig, ax = plt.subplots(figsize=(4, 3))
        # Bỏ emoji 🔴🟢 để tránh lỗi font
        clean_labels = [label.split(' ')[0] for label in labels]
        pie = ax.pie(sizes, autopct='%1.1f%%', colors=colors, 
                     startangle=90, textprops={'color':"white", 'weight':"bold"})
        
        # Thêm chú thích
        ax.legend(pie[0], clean_labels, loc="upper right", bbox_to_anchor=(1.5, 1))
        
        ax.axis('equal') # Đảm bảo biểu đồ tròn
        fig.patch.set_alpha(0.0) # Nền trong suốt
        
        st.pyplot(fig) # Hiển thị biểu đồ

    # --- Biểu đồ 2: Feature Importance (Logic cho cả 3) ---
    with col2:
        model_key = model_name.split(' ')[0].lower() # Lấy 'df', 'svm', 'nb'
        st.markdown(f"#### Top 20 Đặc trưng Ảnh hưởng nhất ({model_name.split(' ')[0]})")

        try:
            model = assets[f'model_{model_key}']
            feature_names = assets['columns']
            
            if model_key == 'df':
                # Decision Forest dùng .feature_importances_
                importances = model.feature_importances_
                df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                df_imp = df_imp.sort_values(by='Importance', ascending=False).head(20)
                
            elif model_key == 'svm':
                # SVM (Linear) dùng .coef_
                if hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_[0]) # Lấy giá trị tuyệt đối
                    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance (abs(coef))': importances})
                    df_imp = df_imp.sort_values(by='Importance (abs(coef))', ascending=False).head(20)
                else:
                    st.warning("Mô hình SVM này không phải 'linear', không thể hiển thị .coef_")
                    return

            elif model_key == 'nb':
                # Naive Bayes dùng feature_log_prob_ (cho Multinomial) hoặc theta_ (cho Gaussian)
                if hasattr(model, 'feature_log_prob_'): # MultinomialNB
                    # So sánh log-prob của lớp 0 (Malware) và lớp 1 (Benign)
                    importance = np.abs(model.feature_log_prob_[0] - model.feature_log_prob_[1])
                elif hasattr(model, 'theta_'): # GaussianNB
                    # So sánh trung bình (mean) của lớp 0 và lớp 1
                    importance = np.abs(model.theta_[0] - model.theta_[1])
                else:
                    raise Exception("Không thể xác định loại Naive Bayes")
                
                df_imp = pd.DataFrame({'Feature': feature_names, 'Importance (Diff)': importance})
                df_imp = df_imp.sort_values(by='Importance (Diff)', ascending=False).head(20)
            
            # Sắp xếp lại để biểu đồ bar chart đẹp hơn (quan trọng nhất ở trên cùng)
            df_imp = df_imp.sort_values(by=df_imp.columns[1], ascending=True)
            st.bar_chart(df_imp.set_index('Feature')) # Hiển thị biểu đồ cột

        except Exception as e:
            st.error(f"Lỗi khi tạo biểu đồ Feature Importance: {e}")

# --- 5. HÀM CHÍNH CHẠY PHÂN TÍCH ---
def run_analysis(assets, model_name, input_data, is_df=False, separator='|'):
    """
    Hàm này nhận dữ liệu đầu vào (từ .exe hoặc .csv),
    tiền xử lý, dự đoán, và hiển thị kết quả.
    """
    try:
        # 5.1. Đọc dữ liệu
        if is_df:
            # Nếu là DataFrame (từ .exe), dùng luôn
            df_input = input_data
            df_display = pd.DataFrame({'Tệp đã tải lên': [f"file_{i+1}" for i in range(len(df_input))]})
        else: 
            # Nếu là tệp CSV, đọc tệp
            # Sửa lỗi: na_values='?' để xử lý tệp Drebin
            df_input = pd.read_csv(input_data, sep=separator, encoding='latin1', low_memory=False, na_values='?')
            
            # Lấy cột định danh để hiển thị (ví dụ: Name, md5)
            display_cols = []
            if 'Name' in df_input.columns: display_cols.append('Name')
            if 'md5' in df_input.columns: display_cols.append('md5')
            if 'hash' in df_input.columns: display_cols.append('hash')
            if not display_cols:
                first_col_name = df_input.columns[0]
                if first_col_name not in assets['columns']:
                     display_cols.append(first_col_name)
                else:
                    df_input['file_id'] = [f"file_{i+1}" for i in range(len(df_input))]
                    display_cols = ['file_id']
            df_display = df_input[display_cols].copy()

        # 5.2. Kiểm tra các cột đặc trưng
        expected_cols = assets['columns']
        missing_cols = [col for col in expected_cols if col not in df_input.columns]
        
        if missing_cols:
            st.error(f"LỖI: Dữ liệu đầu vào thiếu các cột: {', '.join(missing_cols)}")
            return

        X_input_raw = df_input[expected_cols]

        # 5.3. Tiền xử lý (Imputer & Scaler)
        if 'imputer' in assets:
            X_input_imputed = assets['imputer'].transform(X_input_raw)
        else:
            X_input_imputed = X_input_raw
            
        model_key = model_name.split(' ')[0].lower() # df, svm, nb
        
        # Kiểm tra xem mô hình có cần scale không
        needs_scaling = False
        if assets['type'] == 'pe_header': needs_scaling = True # PE Header luôn scale
        elif assets['type'] == 'apk_drebin' and model_key in ['df', 'svm']: needs_scaling = True
        elif assets['type'] == 'api_imports' and model_key == 'svm': needs_scaling = True
            
        if needs_scaling:
            if 'scaler' in assets:
                X_processed = assets['scaler'].transform(X_input_imputed)
            else:
                st.error("Lỗi: Mô hình này cần Scaler nhưng không tìm thấy tệp.")
                return
        else:
            X_processed = X_input_imputed # Dùng dữ liệu đã imputer (nếu có)
            
        # 5.4. Dự đoán
        model = assets[f'model_{model_key}']
        predictions = model.predict(X_processed)
        
        # 5.5. Hiển thị kết quả (trong Thẻ Card)
        with st.container(border=True):
            # Xử lý nhãn cho Drebin (S/B) và PE (0/1)
            if assets['type'] == 'apk_drebin':
                 df_display['Dự đoán (0=S/Malware, 1=B/Benign)'] = predictions
                 df_display['Kết quả'] = np.where(predictions == 0, 'Phát hiện Malware 🔴 (S)', 'An toàn 🟢 (B)')
            else:
                 df_display['Dự đoán (0=Malware, 1=Lành tính)'] = predictions
                 df_display['Kết quả'] = np.where(predictions == 0, 'Phát hiện Malware 🔴', 'An toàn 🟢')

            # Hiển thị độ tin cậy (nếu có)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_processed)
                if assets['type'] == 'apk_drebin':
                     df_display['Độ tin cậy (Malware/S)'] = [f"{p[0]*100:.2f}%" for p in probabilities]
                     df_display['Độ tin cậy (Benign/B)'] = [f"{p[1]*100:.2f}%" for p in probabilities]
                else:
                     df_display['Độ tin cậy (Malware)'] = [f"{p[0]*100:.2f}%" for p in probabilities]
                     df_display['Độ tin cậy (Lành tính)'] = [f"{p[1]*100:.2f}%" for p in probabilities]

            st.markdown(f"<h3>📋 Kết quả phân tích (Sử dụng: {model_name})</h3>", unsafe_allow_html=True)
            st.dataframe(df_display, use_container_width=True)

            # Gọi hàm vẽ biểu đồ
            plot_results(df_display, assets, model_name)

    except pd.errors.ParserError:
        st.error(f"LỖI: Không thể đọc tệp CSV. Bạn có chắc chắn đã sử dụng dấu phân cách là '{separator}' chưa?")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi xử lý tệp: {e}")
        st.exception(e)


# --- 6. GIAO DIỆN CHÍNH (SIDEBAR VÀ NỘI DUNG) ---

# --- Sidebar ---
st.sidebar.title("🔬 Bảng điều khiển")
st.sidebar.write("**Phân tích và phát hiện phần mềm độc hại bằng Machine Learning**")
st.sidebar.info(f"Đang tải mô hình từ: `{MODEL_DIRECTORY}`")

analysis_type = st.sidebar.radio(
    "Chọn loại phân tích:",
    ('🪟 Phân tích PE Header', '📚 Phân tích PE API Imports', '📱 Phân tích APK (Android)')
)

# --- TRANG 1: PHÂN TÍCH PE HEADER ---
if analysis_type == '🪟 Phân tích PE Header':
    st.title("🪟 Phân tích PE Header (Windows .exe)")
    st.write("Sử dụng 54 đặc trưng từ PE Header để phân loại tệp.")
    
    assets = load_assets('pe_header', MODEL_DIRECTORY)
    if assets:
        assets['type'] = 'pe_header'
        
        # Đặt các lựa chọn vào trong một 'Thẻ' (Card)
        with st.container(border=True):
            st.markdown("### 1. Chọn Thuật toán")
            model_name = st.selectbox(
                "Chọn thuật toán bạn muốn sử dụng để dự đoán:",
                ("DF (Decision Forest)", "SVM (Linear)", "NB (Gaussian)"),
                label_visibility="collapsed"
            )
            
            st.markdown("### 2. Tải lên Dữ liệu")
            # Dùng Tabs (Tab) thay vì Radio
            tab1, tab2 = st.tabs(["📁 Tải lên tệp .exe (Tự động)", "📄 Tải lên tệp CSV (Thủ công)"])

            with tab1: # Tab 1: Tải .exe
                st.info("Tải lên tệp `.exe` hoặc `.dll`. Hệ thống sẽ tự động trích xuất 54 đặc trưng và dự đoán.")
                uploaded_file = st.file_uploader("Tải lên tệp PE", type=["exe", "dll"], key="pe_exe_uploader")
                if uploaded_file:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        with st.spinner("Đang trích xuất 54 đặc trưng từ tệp PE..."):
                            df_features = feature_extractor.get_pe_header_features(tmp_file_path, MODEL_DIRECTORY)
                        st.success("Trích xuất thành công. Đang dự đoán...")
                        run_analysis(assets, model_name, df_features, is_df=True)
                    
                    except Exception as e:
                        st.error(f"Lỗi khi trích xuất đặc trưng: {e}")
                    finally:
                        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)

            with tab2: # Tab 2: Tải .csv
                with st.expander("Nhắc lại: Yêu cầu định dạng CSV"):
                    st.info(f"Tải tệp CSV (giống `MalwareData.csv`) có 54 đặc trưng, phân cách bằng dấu `|`.")
                
                uploaded_file_csv = st.file_uploader("Tải tệp PE Header CSV", type=["csv"], key="pe_csv_uploader")
                if uploaded_file_csv:
                    run_analysis(assets, model_name, uploaded_file_csv, is_df=False, separator='|')

# --- TRANG 2: PHÂN TÍCH PE API IMPORTS ---
elif analysis_type == '📚 Phân tích PE API Imports':
    st.title("📚 Phân tích PE API Imports (Windows .exe)")
    st.write("Sử dụng 1000 đặc trưng là tần suất các hàm API được gọi.")
    
    assets = load_assets('api_imports', MODEL_DIRECTORY)
    if assets:
        assets['type'] = 'api_imports'

        with st.container(border=True):
            st.markdown("### 1. Chọn Thuật toán")
            model_name = st.selectbox(
                "Chọn thuật toán bạn muốn sử dụng để dự đoán:",
                ("DF (Decision Forest)", "SVM (Linear)", "NB (Multinomial)"),
                label_visibility="collapsed"
            )
            
            st.markdown("### 2. Tải lên Dữ liệu")
            tab1, tab2 = st.tabs(["📁 Tải lên tệp .exe (Tự động)", "📄 Tải lên tệp CSV (Thủ công)"])
            
            with tab1:
                st.info("Tải lên tệp `.exe` hoặc `.dll`. Hệ thống sẽ tự động trích xuất 1000 đặc trưng API và dự đoán.")
                uploaded_file = st.file_uploader("Tải lên tệp PE", type=["exe", "dll"], key="api_exe_uploader")
                if uploaded_file:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        with st.spinner("Đang trích xuất 1000 đặc trưng API Imports..."):
                            df_features = feature_extractor.get_api_imports_features(tmp_file_path, MODEL_DIRECTORY)
                        st.success("Trích xuất thành công. Đang dự đoán...")
                        run_analysis(assets, model_name, df_features, is_df=True)
                    
                    except Exception as e:
                        st.error(f"Lỗi khi trích xuất đặc trưng: {e}")
                    finally:
                        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)
            
            with tab2:
                with st.expander("Nhắc lại: Yêu cầu định dạng CSV"):
                    st.info(f"Tải tệp CSV (giống `top_1000_pe_imports.csv`), phân cách bằng dấu `,`.")
                
                uploaded_file_csv = st.file_uploader("Tải tệp API Imports CSV", type=["csv"], key="api_csv_uploader")
                if uploaded_file_csv:
                    run_analysis(assets, model_name, uploaded_file_csv, is_df=False, separator=',')

# --- TRANG 3: PHÂN TÍCH APK (ANDROID) ---
elif analysis_type == '📱 Phân tích APK (Android)':
    st.title("📱 Phân tích APK (Bộ dữ liệu Drebin)")
    st.write("Sử dụng 215 đặc trưng (permissions, v.v.) từ bộ dữ liệu Drebin.")
    
    assets = load_assets('apk_drebin', MODEL_DIRECTORY)
    
    if assets:
        assets['type'] = 'apk_drebin'
        
        with st.container(border=True):
            st.markdown("### 1. Chọn Thuật toán")
            model_name = st.selectbox(
                "Chọn thuật toán bạn muốn sử dụng để dự đoán:",
                ("DF (Decision Forest)", "SVM (Linear)", "NB (Multinomial)"),
                label_visibility="collapsed"
            )
            
            st.markdown("### 2. Tải lên Dữ liệu")
            
            st.warning("⚠️ Chức năng trích xuất tự động từ tệp `.apk` đang được phát triển.")
            with st.expander("Giải thích lý do & Yêu cầu định dạng CSV"):
                st.info("""
                Phần này sử dụng mô hình được huấn luyện trên **bộ dữ liệu Drebin (215 đặc trưng)**.
                Hiện tại, dự án chưa hỗ trợ trích xuất 215 đặc trưng này tự động từ tệp `.apk`.
                
                **Yêu cầu:** Vui lòng chỉ tải lên tệp CSV (giống `drebin-215...csv`), phân cách bằng dấu `,`. Tệp này có chứa ký tự `?` và sẽ được tự động xử lý.
                """)
            
            uploaded_file_csv = st.file_uploader("Tải tệp APK (Drebin) CSV", type=["csv"], key="apk_csv_uploader")
            
            if uploaded_file_csv:
                run_analysis(assets, model_name, uploaded_file_csv, is_df=False, separator=',')