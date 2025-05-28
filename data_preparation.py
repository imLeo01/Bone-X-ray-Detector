import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Đường dẫn cơ sở
base_dir = r'C:\Users\Admin\Downloads\fonai\MURA'

# Danh sách các vùng X-quang
regions = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

# Tạo thư mục cho cấu trúc phân chia 3 tập
for region in regions:
    # Tạo thư mục tập train mới
    os.makedirs(os.path.join(base_dir, f'mura_final/{region}/train/normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'mura_final/{region}/train/abnormal'), exist_ok=True)
    
    # Tạo thư mục tập validation 
    os.makedirs(os.path.join(base_dir, f'mura_final/{region}/val/normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'mura_final/{region}/val/abnormal'), exist_ok=True)
    
    # Tạo thư mục tập test (sử dụng valid set gốc)
    os.makedirs(os.path.join(base_dir, f'mura_final/{region}/test/normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f'mura_final/{region}/test/abnormal'), exist_ok=True)

# Hàm để đếm số lượng ảnh trong thư mục
def count_images(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

# Phân chia và sao chép dữ liệu
total_train_images = 0
total_val_images = 0
total_test_images = 0

print("Bắt đầu chia dữ liệu thành 3 tập...")

for region in regions:
    print(f"\nĐang xử lý vùng {region}...")
    
    for condition in ['normal', 'abnormal']:
        # Lấy đường dẫn thư mục gốc
        train_original_dir = os.path.join(base_dir, f'mura_classified/{region}/train/{condition}')
        valid_original_dir = os.path.join(base_dir, f'mura_classified/{region}/valid/{condition}')
        
        # Lấy danh sách tất cả các file ảnh từ tập train gốc
        train_images = []
        if os.path.exists(train_original_dir):
            train_images = [f for f in os.listdir(train_original_dir) 
                           if os.path.isfile(os.path.join(train_original_dir, f))]
        
        # Chia tập train gốc thành train mới (80%) và val (20%)
        if train_images:
            train_new, val_new = train_test_split(train_images, test_size=0.2, random_state=42)
            
            # Sao chép ảnh vào thư mục train mới
            for img in train_new:
                src_path = os.path.join(train_original_dir, img)
                dst_path = os.path.join(base_dir, f'mura_final/{region}/train/{condition}', img)
                shutil.copy2(src_path, dst_path)
                total_train_images += 1
            
            # Sao chép ảnh vào thư mục validation
            for img in val_new:
                src_path = os.path.join(train_original_dir, img)
                dst_path = os.path.join(base_dir, f'mura_final/{region}/val/{condition}', img)
                shutil.copy2(src_path, dst_path)
                total_val_images += 1
        
        # Sao chép tất cả ảnh từ valid set gốc vào test set mới
        if os.path.exists(valid_original_dir):
            valid_images = [f for f in os.listdir(valid_original_dir) 
                           if os.path.isfile(os.path.join(valid_original_dir, f))]
            
            for img in valid_images:
                src_path = os.path.join(valid_original_dir, img)
                dst_path = os.path.join(base_dir, f'mura_final/{region}/test/{condition}', img)
                shutil.copy2(src_path, dst_path)
                total_test_images += 1

print(f"\nĐã hoàn thành phân chia dữ liệu!")
print(f"Tổng số ảnh trong tập train: {total_train_images}")
print(f"Tổng số ảnh trong tập validation: {total_val_images}")
print(f"Tổng số ảnh trong tập test: {total_test_images}")

# Thống kê chi tiết theo từng vùng
print("\nThống kê chi tiết theo từng vùng X-quang:")
for region in regions:
    # Đếm số lượng ảnh trong mỗi tập
    train_normal_count = count_images(os.path.join(base_dir, f'mura_final/{region}/train/normal'))
    train_abnormal_count = count_images(os.path.join(base_dir, f'mura_final/{region}/train/abnormal'))
    val_normal_count = count_images(os.path.join(base_dir, f'mura_final/{region}/val/normal'))
    val_abnormal_count = count_images(os.path.join(base_dir, f'mura_final/{region}/val/abnormal'))
    test_normal_count = count_images(os.path.join(base_dir, f'mura_final/{region}/test/normal'))
    test_abnormal_count = count_images(os.path.join(base_dir, f'mura_final/{region}/test/abnormal'))
    
    # Tính tổng số ảnh cho mỗi vùng
    total_region_train = train_normal_count + train_abnormal_count
    total_region_val = val_normal_count + val_abnormal_count
    total_region_test = test_normal_count + test_abnormal_count
    total_region = total_region_train + total_region_val + total_region_test
    
    # In thống kê
    print(f"\n=== {region} ===")
    print(f"Train set: {train_normal_count} normal, {train_abnormal_count} abnormal (tổng: {total_region_train})")
    print(f"Val set: {val_normal_count} normal, {val_abnormal_count} abnormal (tổng: {total_region_val})")
    print(f"Test set: {test_normal_count} normal, {test_abnormal_count} abnormal (tổng: {total_region_test})")
    print(f"Tổng cộng: {total_region} ảnh")
    
    # Tính tỷ lệ normal/abnormal
    if total_region_train > 0:
        train_normal_ratio = train_normal_count / total_region_train * 100
        train_abnormal_ratio = train_abnormal_count / total_region_train * 100
        print(f"Tỷ lệ train: {train_normal_ratio:.1f}% normal, {train_abnormal_ratio:.1f}% abnormal")
    
    if total_region_val > 0:
        val_normal_ratio = val_normal_count / total_region_val * 100
        val_abnormal_ratio = val_abnormal_count / total_region_val * 100
        print(f"Tỷ lệ val: {val_normal_ratio:.1f}% normal, {val_abnormal_ratio:.1f}% abnormal")
    
    if total_region_test > 0:
        test_normal_ratio = test_normal_count / total_region_test * 100
        test_abnormal_ratio = test_abnormal_count / total_region_test * 100
        print(f"Tỷ lệ test: {test_normal_ratio:.1f}% normal, {test_abnormal_ratio:.1f}% abnormal")

print("\nChú ý: Dữ liệu đã được phân chia theo tiêu chuẩn học máy với:")
print("- Train set (80% dữ liệu train gốc): Để huấn luyện mô hình")
print("- Validation set (20% dữ liệu train gốc): Để điều chỉnh mô hình và tránh overfitting")
print("- Test set (Giữ nguyên valid set gốc): Để đánh giá hiệu suất cuối cùng")