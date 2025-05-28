import pandas as pd
import os
from PIL import Image
import numpy as np

# Đường dẫn tới thư mục MURA (đã điều chỉnh)
mura_dir = r'C:\Users\Admin\Downloads\fonai\MURA'


# Đọc tệp CSV (được tìm thấy trực tiếp trong thư mục MURA)
train_labels_path = os.path.join(mura_dir, 'train_labeled_studies.csv')
valid_labels_path = os.path.join(mura_dir, 'valid_labeled_studies.csv')

# Kiểm tra xem file có tồn tại không
if not os.path.exists(train_labels_path):
    print(f"File {train_labels_path} không tồn tại!")
if not os.path.exists(valid_labels_path):
    print(f"File {valid_labels_path} không tồn tại!")

# Kiểm tra thư mục
print(f"Thư mục MURA có tồn tại: {os.path.exists(mura_dir)}")
if os.path.exists(mura_dir):
    print("Các mục trong thư mục MURA:")
    for item in os.listdir(mura_dir):
        print(f"- {item}")

# Danh sách các vùng cơ thể trong bộ dữ liệu MURA
regions = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

# Thư mục cơ sở
base_dir = r'C:\Users\Admin\Downloads\fonai\MURA'

# Tạo cấu trúc thư mục cho từng vùng cơ thể
for region in regions:
    os.makedirs(os.path.join(base_dir, f'mura_classified/{region}/train/normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'mura_classified/{region}/train/abnormal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'mura_classified/{region}/valid/normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'mura_classified/{region}/valid/abnormal'), exist_ok=True)

# Chỉ tiếp tục nếu file CSV tồn tại
if os.path.exists(train_labels_path) and os.path.exists(valid_labels_path):
    train_labels = pd.read_csv(train_labels_path, header=None)
    valid_labels = pd.read_csv(valid_labels_path, header=None)

    # Kiểm tra dữ liệu CSV
    print("\nDữ liệu train (5 dòng đầu):")
    print(train_labels.head())
    print("\nDữ liệu valid (5 dòng đầu):")
    print(valid_labels.head())
    
    # Kiểm tra một số đường dẫn
    print("\nKiểm tra một số đường dẫn từ CSV:")
    for i in range(min(5, len(train_labels))):
        csv_path = train_labels.iloc[i, 0]
        full_path = os.path.join(mura_dir, csv_path)
        print(f"Đường dẫn: {full_path}")
        print(f"Tồn tại: {os.path.exists(full_path)}")
        
        # Nếu không tồn tại, kiểm tra các biến thể đường dẫn
        if not os.path.exists(full_path):
            # Loại bỏ "MURA-v1.1" nếu có trong đường dẫn
            modified_path = full_path.replace('MURA-v1.1\\', '').replace('MURA-v1.1/', '')
            print(f"Đường dẫn đã sửa: {modified_path}")
            print(f"Tồn tại: {os.path.exists(modified_path)}")
    
    # Tổ chức ảnh train
    print("\nĐang tổ chức ảnh train...")
    processed_count = 0
    error_count = 0
    
    for index, row in train_labels.iterrows():
        csv_path = row[0]
        label = row[1]
        
        # Chuẩn hóa đường dẫn (đảm bảo dùng dấu \ cho Windows)
        csv_path = csv_path.replace('/', '\\')
        
        # Loại bỏ "MURA-v1.1" nếu có trong đường dẫn
        if "MURA-v1.1" in csv_path:
            csv_path = csv_path.replace('MURA-v1.1\\', '').replace('MURA-v1.1/', '')
        
        full_path = os.path.join(mura_dir, csv_path)
        
        if not os.path.exists(full_path):
            error_count += 1
            if error_count <= 10:  # Chỉ hiển thị 10 lỗi đầu tiên
                print(f"Thư mục không tồn tại: {full_path}")
            continue
        
        # Xác định vùng cơ thể từ đường dẫn
        region = None
        for r in regions:
            if r in csv_path:
                region = r
                break
        
        if region is None:
            print(f"Không thể xác định vùng cơ thể từ {csv_path}")
            continue
        
        dst_dir = os.path.join(base_dir, f'mura_classified/{region}/train', 'abnormal' if label == 1 else 'normal')
        
        # Sao chép ảnh
        try:
            for image_file in os.listdir(full_path):
                src_path = os.path.join(full_path, image_file)
                if not os.path.isfile(src_path):
                    continue
                
                # Lấy thông tin patient và study từ đường dẫn
                path_parts = full_path.split('\\')
                patient_id = None
                study_id = None
                
                for part in path_parts:
                    if part.startswith('patient'):
                        patient_id = part
                    elif part.startswith('study'):
                        study_id = part
                
                if not patient_id or not study_id:
                    # Nếu không tìm thấy, sử dụng 2 thư mục cuối cùng
                    if len(path_parts) >= 2:
                        patient_id = path_parts[-2]
                        study_id = path_parts[-1]
                    else:
                        patient_id = "unknown_patient"
                        study_id = "unknown_study"
                
                dst_path = os.path.join(dst_dir, f"{patient_id}_{study_id}_{image_file}")
                
                # Đọc và lưu ảnh dưới dạng grayscale
                try:
                    img = Image.open(src_path).convert('L')
                    img_array = np.array(img)
                    img = Image.fromarray(img_array)
                    img.save(dst_path)
                    processed_count += 1
                    if processed_count % 100 == 0:  # In thông báo mỗi 100 ảnh
                        print(f"Đã xử lý {processed_count} ảnh")
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {src_path}: {e}")
        except Exception as e:
            print(f"Lỗi khi liệt kê thư mục {full_path}: {e}")
    
    print(f"Đã xử lý tổng cộng {processed_count} ảnh train, có {error_count} lỗi")
    
    # Tổ chức ảnh valid
    print("\nĐang tổ chức ảnh valid...")
    processed_count = 0
    error_count = 0
    
    for index, row in valid_labels.iterrows():
        csv_path = row[0]
        label = row[1]
        
        # Chuẩn hóa đường dẫn
        csv_path = csv_path.replace('/', '\\')
        
        # Loại bỏ "MURA-v1.1" nếu có trong đường dẫn
        if "MURA-v1.1" in csv_path:
            csv_path = csv_path.replace('MURA-v1.1\\', '').replace('MURA-v1.1/', '')
        
        full_path = os.path.join(mura_dir, csv_path)
        
        if not os.path.exists(full_path):
            error_count += 1
            if error_count <= 10:  # Chỉ hiển thị 10 lỗi đầu tiên
                print(f"Thư mục không tồn tại: {full_path}")
            continue
        
        # Xác định vùng cơ thể từ đường dẫn
        region = None
        for r in regions:
            if r in csv_path:
                region = r
                break
        
        if region is None:
            print(f"Không thể xác định vùng cơ thể từ {csv_path}")
            continue
        
        dst_dir = os.path.join(base_dir, f'mura_classified/{region}/valid', 'abnormal' if label == 1 else 'normal')
        
        # Sao chép ảnh
        try:
            for image_file in os.listdir(full_path):
                src_path = os.path.join(full_path, image_file)
                if not os.path.isfile(src_path):
                    continue
                
                # Lấy thông tin patient và study từ đường dẫn
                path_parts = full_path.split('\\')
                patient_id = None
                study_id = None
                
                for part in path_parts:
                    if part.startswith('patient'):
                        patient_id = part
                    elif part.startswith('study'):
                        study_id = part
                
                if not patient_id or not study_id:
                    # Nếu không tìm thấy, sử dụng 2 thư mục cuối cùng
                    if len(path_parts) >= 2:
                        patient_id = path_parts[-2]
                        study_id = path_parts[-1]
                    else:
                        patient_id = "unknown_patient"
                        study_id = "unknown_study"
                
                dst_path = os.path.join(dst_dir, f"{patient_id}_{study_id}_{image_file}")
                
                # Đọc và lưu ảnh dưới dạng grayscale
                try:
                    img = Image.open(src_path).convert('L')
                    img_array = np.array(img)
                    img = Image.fromarray(img_array)
                    img.save(dst_path)
                    processed_count += 1
                    if processed_count % 100 == 0:  # In thông báo mỗi 100 ảnh
                        print(f"Đã xử lý {processed_count} ảnh")
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {src_path}: {e}")
        except Exception as e:
            print(f"Lỗi khi liệt kê thư mục {full_path}: {e}")
    
    print(f"Đã xử lý tổng cộng {processed_count} ảnh valid, có {error_count} lỗi")
    
    # Kiểm tra số lượng ảnh cho mỗi vùng
    print("\nKết quả thống kê:")
    for region in regions:
        train_normal_dir = os.path.join(base_dir, f'mura_classified/{region}/train/normal')
        train_abnormal_dir = os.path.join(base_dir, f'mura_classified/{region}/train/abnormal')
        valid_normal_dir = os.path.join(base_dir, f'mura_classified/{region}/valid/normal')
        valid_abnormal_dir = os.path.join(base_dir, f'mura_classified/{region}/valid/abnormal')
        
        print(f"\nThống kê cho vùng {region}:")
        print(f"Số lượng ảnh train normal: {len(os.listdir(train_normal_dir)) if os.path.exists(train_normal_dir) else 0}")
        print(f"Số lượng ảnh train abnormal: {len(os.listdir(train_abnormal_dir)) if os.path.exists(train_abnormal_dir) else 0}")
        print(f"Số lượng ảnh valid normal: {len(os.listdir(valid_normal_dir)) if os.path.exists(valid_normal_dir) else 0}")
        print(f"Số lượng ảnh valid abnormal: {len(os.listdir(valid_abnormal_dir)) if os.path.exists(valid_abnormal_dir) else 0}")
else:
    print("Không thể tiếp tục vì file CSV không tồn tại")