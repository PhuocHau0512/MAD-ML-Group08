import pefile # Thư viện chính để đọc tệp .exe
import pandas as pd
import joblib
import os

def get_pe_header_features(file_path, model_dir):
    """
    Trích xuất 54 đặc trưng PE Header từ một tệp .exe.
    Đọc tệp danh sách cột (đã lưu) từ model_dir để đảm bảo trật tự.
    """
    
    # 1. Tải danh sách 54 cột đặc trưng mà mô hình đã học
    columns_path = os.path.join(model_dir, 'pe_header_columns.joblib')
    try:
        columns = joblib.load(columns_path)
    except FileNotFoundError:
        raise Exception(f"Không tìm thấy tệp '{columns_path}'. Vui lòng chạy train_all_models.py")
        
    # 2. Tạo một "vector 0" (dictionary) với 54 cột
    features = {col: 0 for col in columns}
    
    # 3. Mở tệp .exe và trích xuất đặc trưng
    pe = None # Khởi tạo pe là None
    try:
        pe = pefile.PE(file_path)
        
        # --- Trích xuất các đặc trưng từ PE Header ---
        
        # Đặc trưng từ FILE_HEADER
        features['Machine'] = pe.FILE_HEADER.Machine
        features['SizeOfOptionalHeader'] = pe.FILE_HEADER.SizeOfOptionalHeader
        features['Characteristics'] = pe.FILE_HEADER.Characteristics
        
        # Đặc trưng từ OPTIONAL_HEADER
        features['MajorLinkerVersion'] = pe.OPTIONAL_HEADER.MajorLinkerVersion
        features['MinorLinkerVersion'] = pe.OPTIONAL_HEADER.MinorLinkerVersion
        features['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
        features['SizeOfInitializedData'] = pe.OPTIONAL_HEADER.SizeOfInitializedData
        features['SizeOfUninitializedData'] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
        features['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        features['BaseOfCode'] = pe.OPTIONAL_HEADER.BaseOfCode
        features['BaseOfData'] = pe.OPTIONAL_HEADER.BaseOfData
        features['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
        features['SectionAlignment'] = pe.OPTIONAL_HEADER.SectionAlignment
        features['FileAlignment'] = pe.OPTIONAL_HEADER.FileAlignment
        features['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
        features['MinorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
        features['MajorImageVersion'] = pe.OPTIONAL_HEADER.MajorImageVersion
        features['MinorImageVersion'] = pe.OPTIONAL_HEADER.MinorImageVersion
        features['MajorSubsystemVersion'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
        features['MinorSubsystemVersion'] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
        features['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
        features['SizeOfHeaders'] = pe.OPTIONAL_HEADER.SizeOfHeaders
        features['CheckSum'] = pe.OPTIONAL_HEADER.CheckSum
        features['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
        features['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
        features['SizeOfStackReserve'] = pe.OPTIONAL_HEADER.SizeOfStackReserve
        features['SizeOfStackCommit'] = pe.OPTIONAL_HEADER.SizeOfStackCommit
        features['SizeOfHeapReserve'] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
        features['SizeOfHeapCommit'] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
        features['LoaderFlags'] = pe.OPTIONAL_HEADER.LoaderFlags
        features['NumberOfRvaAndSizes'] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
        
        # Đặc trưng về Sections (tính toán)
        features['SectionsNb'] = len(pe.sections)
        if features['SectionsNb'] > 0:
            entropies = [s.get_entropy() for s in pe.sections if s.get_entropy() > 0]
            raw_sizes = [s.SizeOfRawData for s in pe.sections]
            virtual_sizes = [s.Misc_VirtualSize for s in pe.sections]
            
            if entropies:
                features['SectionsMeanEntropy'] = sum(entropies) / len(entropies)
                features['SectionsMinEntropy'] = min(entropies)
                features['SectionsMaxEntropy'] = max(entropies)
            if raw_sizes:
                features['SectionsMeanRawsize'] = sum(raw_sizes) / len(raw_sizes)
                features['SectionsMinRawsize'] = min(raw_sizes)
                features['SectionMaxRawsize'] = max(raw_sizes)
            if virtual_sizes:
                features['SectionsMeanVirtualsize'] = sum(virtual_sizes) / len(virtual_sizes)
                features['SectionsMinVirtualsize'] = min(virtual_sizes)
                features['SectionMaxVirtualsize'] = max(virtual_sizes)

        # Đặc trưng về Imports (Hàm API)
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            features['ImportsNbDLL'] = len(pe.DIRECTORY_ENTRY_IMPORT)
            features['ImportsNb'] = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT)
            features['ImportsNbOrdinal'] = sum(1 for entry in pe.DIRECTORY_ENTRY_IMPORT for imp in entry.imports if imp.name is None)
        
        # Đặc trưng Exports
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            features['ExportNb'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
            
        # Đặc trưng Resources
        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
            features['ResourcesNb'] = len(pe.DIRECTORY_ENTRY_RESOURCE.entries)
            try:
                r_entropies, r_sizes = [], []
                for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                    if hasattr(entry, 'directory') and entry.directory.entries:
                        for sub_entry in entry.directory.entries:
                             if hasattr(sub_entry, 'data') and hasattr(sub_entry.data, 'struct'):
                                entropy = sub_entry.data.struct.get_entropy()
                                if entropy > 0: r_entropies.append(entropy)
                                r_sizes.append(sub_entry.data.struct.Size)
                if r_entropies:
                    features['ResourcesMeanEntropy'] = sum(r_entropies) / len(r_entropies)
                    features['ResourcesMinEntropy'] = min(r_entropies)
                    features['ResourcesMaxEntropy'] = max(r_entropies)
                if r_sizes:
                    features['ResourcesMeanSize'] = sum(r_sizes) / len(r_sizes)
                    features['ResourcesMinSize'] = min(r_sizes)
                    features['ResourcesMaxSize'] = max(r_sizes)
            except Exception: pass # Bỏ qua nếu không trích xuất được

        # Đặc trưng LoadConfiguration
        if hasattr(pe, 'DIRECTORY_ENTRY_LOAD_CONFIG'):
            features['LoadConfigurationSize'] = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
        
        # Đặc trưng VersionInformation
        if hasattr(pe, 'VS_VERSIONINFO'):
            if hasattr(pe, 'VS_FIXEDFILEINFO'):
                features['VersionInformationSize'] = pe.VS_FIXEDFILEINFO[0].sizeof()
                
    except pefile.PEFormatError:
        print(f"Lỗi: {file_path} không phải là tệp PE hợp lệ.")
    except Exception as e:
        print(f"Lỗi khi trích xuất PE Header: {e}")
        
    # SỬA LỖI WinError 32: Đảm bảo tệp pefile luôn được đóng
    finally:
        if pe:
            pe.close()
            
    # 4. Trả về DataFrame với 1 hàng duy nhất
    return pd.DataFrame([features])[columns]

def get_api_imports_features(file_path, model_dir):
    """
    Trích xuất 1000 đặc trưng API Imports (tần suất) từ một tệp .exe.
    """
    
    # 1. Tải 1000 tên cột API mà mô hình đã học
    columns_path = os.path.join(model_dir, 'api_imports_columns.joblib')
    try:
        columns = joblib.load(columns_path)
    except FileNotFoundError:
        raise Exception(f"Không tìm thấy tệp '{columns_path}'. Vui lòng chạy train_all_models.py")
        
    # 2. Tạo một "vector 0" (dictionary) với 1000 cột
    features = {col: 0 for col in columns}
    
    # 3. Mở tệp .exe và đếm các hàm API
    pe = None # Khởi tạo pe là None
    try:
        pe = pefile.PE(file_path)
        
        # Kiểm tra xem tệp có Bảng Import không
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            # Duyệt qua từng DLL (ví dụ: kernel32.dll)
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                # Duyệt qua từng hàm (API) được import từ DLL đó
                for imp in entry.imports:
                    if imp.name: # Chỉ lấy các hàm có tên
                        api_name = imp.name.decode('utf-8')
                        # Nếu tên hàm nằm trong 1000 cột của chúng ta, đếm nó
                        if api_name in features:
                            features[api_name] += 1
                            
    except pefile.PEFormatError:
        print(f"Lỗi: {file_path} không phải là tệp PE hợp lệ.")
    except Exception as e:
        print(f"Lỗi khi trích xuất API Imports: {e}")
        
    # SỬA LỖI WinError 32: Đảm bảo tệp pefile luôn được đóng
    finally:
        if pe:
            pe.close()
            
    # 4. Trả về DataFrame với 1 hàng duy nhất
    return pd.DataFrame([features])[columns]