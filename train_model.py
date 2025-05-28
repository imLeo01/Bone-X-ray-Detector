# train_model.py
# File này kết hợp định nghĩa mô hình và huấn luyện

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import datetime
from tensorflow.keras.applications import DenseNet121, ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Kiểm tra phiên bản
print(f"TensorFlow version: {tf.__version__}")

# Khai báo các đường dẫn
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Tạo các mô hình
def create_densenet121_model(input_shape=(224, 224, 3), weights='imagenet'):
    """Tạo mô hình DenseNet121 với input RGB"""
    # Base model
    base_model = DenseNet121(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # Đóng băng các lớp
    for layer in base_model.layers:
        layer.trainable = False
    
    # Tạo mô hình tuần tự
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    return model, base_model

def create_resnet50v2_model(input_shape=(224, 224, 3), weights='imagenet'):
    """Tạo mô hình ResNet50V2 với input RGB"""
    # Base model
    base_model = ResNet50V2(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # Đóng băng các lớp
    for layer in base_model.layers:
        layer.trainable = False
    
    # Tạo mô hình tuần tự
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    return model, base_model

def create_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Tạo và lưu confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # Thêm text vào các ô
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()

def create_roc_curve(y_true, y_pred_prob, title, filename):
    """Tạo và lưu đường cong ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

# Callback tùy chỉnh để lưu thông tin epoch
class SaveEpochInfo(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(SaveEpochInfo, self).__init__()
        self.filepath = filepath
        
    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, 'w') as f:
            json.dump({
                'epoch': epoch + 1,
                'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'val_loss': float(logs.get('val_loss', 0)),
                'val_accuracy': float(logs.get('val_accuracy', 0)),
                'val_auc': float(logs.get('val_auc', 0))
            }, f)

def train_model(base_dir, target_region, model_name, img_size=(224, 224), batch_size=32, epochs=50, initial_epoch=0, continue_model=None):
    """Huấn luyện mô hình"""
    print("\n" + "="*50)
    print(f"{'TIẾP TỤC' if initial_epoch > 0 else 'BẮT ĐẦU'} HUẤN LUYỆN MÔ HÌNH {model_name.upper()} CHO VÙNG {target_region}")
    if initial_epoch > 0:
        print(f"Tiếp tục từ epoch {initial_epoch}/{epochs}")
    print("="*50)
    
    # Kiểm tra nếu base_dir không phải là đường dẫn tuyệt đối
    if not os.path.isabs(base_dir):
        print(f"Đường dẫn đầu vào: {base_dir}")
        # Giả sử mura_final nằm trong thư mục MURA
        base_dir = os.path.join("C:\\Users\\Admin\\Downloads\\fonai\\MURA", base_dir)
        print(f"Đã điều chỉnh đường dẫn thành: {base_dir}")
    
    # Đường dẫn dữ liệu
    train_dir = os.path.join(base_dir, target_region, 'train')
    val_dir = os.path.join(base_dir, target_region, 'val')
    test_dir = os.path.join(base_dir, target_region, 'test')
    
    # In đường dẫn đầy đủ để debug
    print(f"\nĐƯỜNG DẪN ĐẦY ĐỦ:")
    print(f"- Train: {train_dir}")
    print(f"- Validation: {val_dir}")   
    print(f"- Test: {test_dir}")
    
    # Kiểm tra thư mục và hiển thị thông tin chi tiết
    missing_dirs = []
    for directory_type, directory_path in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if os.path.exists(directory_path):
            print(f"Thư mục {directory_type} tồn tại: ✓")
            # Kiểm tra các thư mục con
            abnormal_dir = os.path.join(directory_path, 'abnormal')
            normal_dir = os.path.join(directory_path, 'normal')
            
            if os.path.exists(abnormal_dir) and os.path.exists(normal_dir):
                print(f"  - Thư mục 'abnormal' và 'normal' tồn tại: ✓")
                # Đếm số lượng file trong mỗi thư mục
                abnormal_files = len([f for f in os.listdir(abnormal_dir) if os.path.isfile(os.path.join(abnormal_dir, f))])
                normal_files = len([f for f in os.listdir(normal_dir) if os.path.isfile(os.path.join(normal_dir, f))])
                print(f"  - Số lượng ảnh: abnormal={abnormal_files}, normal={normal_files}")
            else:
                if not os.path.exists(abnormal_dir):
                    print(f"  - Thư mục 'abnormal' KHÔNG tồn tại trong {directory_type}: ✗")
                    missing_dirs.append(abnormal_dir)
                if not os.path.exists(normal_dir):
                    print(f"  - Thư mục 'normal' KHÔNG tồn tại trong {directory_type}: ✗")
                    missing_dirs.append(normal_dir)
        else:
            print(f"Thư mục {directory_type} KHÔNG tồn tại: ✗")
            missing_dirs.append(directory_path)
            # Kiểm tra xem thư mục cha có tồn tại không
            parent_dir = os.path.dirname(directory_path)
            if os.path.exists(parent_dir):
                print(f"  - Thư mục cha '{os.path.basename(parent_dir)}' tồn tại: ✓")
                # Liệt kê các thư mục con trong thư mục cha
                sub_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                print(f"  - Các thư mục con có sẵn: {', '.join(sub_dirs) if sub_dirs else 'Không có'}")
            else:
                print(f"  - Thư mục cha '{os.path.basename(parent_dir)}' KHÔNG tồn tại: ✗")
    
    if missing_dirs:
        print("\nTHƯ MỤC KHÔNG TỒN TẠI:")
        for dir_path in missing_dirs:
            print(f" - {dir_path}")
        
        print("\nĐƯỜNG DẪN HIỆN TẠI:", os.getcwd())
        print(f"\nCẤU TRÚC CHÍNH XÁC CẦN CÓ:")
        print(f"{base_dir}\\{target_region}\\train\\abnormal")
        print(f"{base_dir}\\{target_region}\\train\\normal")
        print(f"{base_dir}\\{target_region}\\val\\abnormal")
        print(f"{base_dir}\\{target_region}\\val\\normal")
        print(f"{base_dir}\\{target_region}\\test\\abnormal")
        print(f"{base_dir}\\{target_region}\\test\\normal")
        
        print("\nVui lòng kiểm tra lại đường dẫn hoặc tạo các thư mục trên.")
        raise ValueError(f"Không thể tiếp tục vì thiếu {len(missing_dirs)} thư mục.")
    
    # Tiếp tục nếu các thư mục đều tồn tại
    print("\nTất cả các thư mục đều tồn tại, tiếp tục huấn luyện...\n")
    
    # Data augmentation cho tập train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Chỉ rescale cho validation và test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        # Tạo generators với color_mode='rgb'
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True,
            color_mode='rgb'  # Thay đổi từ grayscale sang rgb
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            color_mode='rgb'  # Thay đổi từ grayscale sang rgb
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            color_mode='rgb'  # Thay đổi từ grayscale sang rgb
        )
    except Exception as e:
        print(f"\nLỖI KHI TẠO DATA GENERATORS: {str(e)}")
        print("Vui lòng kiểm tra lại cấu trúc thư mục và định dạng ảnh.")
        return None, None
    
    # Thông tin về dữ liệu
    print(f"Tổng số mẫu train: {train_generator.samples}")
    print(f"Tổng số mẫu validation: {validation_generator.samples}")
    print(f"Tổng số mẫu test: {test_generator.samples}")
    print(f"Mapping các lớp: {train_generator.class_indices}")
    
    # Tính class weights
    total_samples = train_generator.samples
    n_positive = np.sum(train_generator.classes)
    n_negative = total_samples - n_positive
    
    weight_for_0 = (1 / n_negative) * (total_samples / 2.0)
    weight_for_1 = (1 / n_positive) * (total_samples / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Class weights: {class_weight}")
    
    # Tạo hoặc nạp lại mô hình
    if continue_model is not None:
        print(f"Tiếp tục huấn luyện mô hình đã có từ epoch {initial_epoch}...")
        model = continue_model
        base_model = None  # Không cần base_model khi tiếp tục huấn luyện
    else:
        # Tạo mô hình với input RGB
        print(f"Đang tạo mô hình {model_name} mới...")
        try:
            if model_name.lower() == 'densenet121':
                model, base_model = create_densenet121_model(input_shape=(*img_size, 3))
            elif model_name.lower() == 'resnet50v2':
                model, base_model = create_resnet50v2_model(input_shape=(*img_size, 3))
            else:
                raise ValueError(f"Không hỗ trợ mô hình: {model_name}")
        except Exception as e:
            print(f"\nLỖI KHI TẠO MÔ HÌNH: {str(e)}")
            return None, None
        
        # Biên dịch mô hình (chỉ khi tạo mô hình mới)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    # Hiển thị tóm tắt mô hình
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_path = f"models/{model_name}_{target_region}_best.h5"
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Callback lưu thông tin tiến trình
    checkpoint_info_path = f"models/{model_name}_{target_region}_checkpoint_info.json"
    save_epoch_info = SaveEpochInfo(checkpoint_info_path)
    
    # Danh sách callbacks
    callbacks = [early_stopping, checkpoint, reduce_lr, save_epoch_info]
    
    # Huấn luyện mô hình (phase 1)
    print(f"\nBắt đầu huấn luyện (phase 1) từ epoch {initial_epoch}...")
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,  # Bắt đầu từ epoch đã có
            callbacks=callbacks,
            class_weight=class_weight
        )
    except Exception as e:
        print(f"\nLỖI TRONG QUÁ TRÌNH HUẤN LUYỆN PHASE 1: {str(e)}")
        return model, None
    
    # Bỏ qua phase 2 (fine-tuning) tạm thời để kiểm tra xem phase 1 có chạy được không
    
    # Lưu mô hình cuối cùng
    final_model_path = f"models/{model_name}_{target_region}_final.h5"
    try:
        model.save(final_model_path)
        print(f"Đã lưu mô hình tại {final_model_path}")
    except Exception as e:
        print(f"Lỗi khi lưu mô hình: {str(e)}")
    
    # Đánh giá trên tập test
    print("\nĐánh giá trên tập test:")
    try:
        test_results = model.evaluate(test_generator)
        metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
        print("Test Results:")
        for name, value in zip(metric_names, test_results):
            print(f"{name}: {value:.4f}")
        
        # Vẽ biểu đồ accuracy và loss
        plt.figure(figsize=(12, 5))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model Accuracy - {model_name} - {target_region}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss - {model_name} - {target_region}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'results/training_history_{model_name}_{target_region}.png')
        plt.close()
        
        # Dự đoán và tạo confusion matrix
        y_pred = model.predict(test_generator)
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_true = test_generator.classes
        
        # Tạo confusion matrix
        create_confusion_matrix(
            y_true, 
            y_pred_classes, 
            ['Normal', 'Abnormal'],
            f'Confusion Matrix - {model_name} - {target_region}',
            f'results/confusion_matrix_{model_name}_{target_region}.png'
        )
        
        # Tạo ROC curve
        create_roc_curve(
            y_true, 
            y_pred,
            f'ROC Curve - {model_name} - {target_region}',
            f'results/roc_curve_{model_name}_{target_region}.png'
        )
    except Exception as e:
        print(f"\nLỖI TRONG QUÁ TRÌNH ĐÁNH GIÁ MÔ HÌNH: {str(e)}")
    
    print(f"\nĐã hoàn thành huấn luyện và đánh giá mô hình {model_name} cho vùng {target_region}")
    return model, history

def train_all_regions(model_name, batch_size, epochs, base_dir):
    """Huấn luyện mô hình cho các vùng chưa hoàn thành hoặc bị gián đoạn"""

    regions = [
        "XR_HAND", "XR_WRIST", "XR_ELBOW", "XR_FINGER", 
        "XR_FOREARM", "XR_HUMERUS", "XR_SHOULDER"
    ]
    
    results = {}
    for region in regions:
        model_path = f"models/{model_name}_{region}_best.h5"
        results_path = f"results/roc_curve_{model_name}_{region}.png"
        final_model_path = f"models/{model_name}_{region}_final.h5"
        checkpoint_info_path = f"models/{model_name}_{region}_checkpoint_info.json"

        # Kiểm tra nếu đã hoàn thành đầy đủ
        if os.path.exists(model_path) and os.path.exists(results_path) and os.path.exists(final_model_path):
            print(f"✅ BỎ QUA: {region} (đã hoàn thành đầy đủ)")
            continue  # Skip vùng đã huấn luyện và có kết quả đầy đủ
        
        # Kiểm tra nếu có checkpoint để tiếp tục
        initial_epoch = 0
        continue_model = None
        
        if os.path.exists(model_path) and os.path.exists(checkpoint_info_path):
            try:
                with open(checkpoint_info_path, 'r') as f:
                    info = json.load(f)
                    initial_epoch = info.get('epoch', 0)
                
                if initial_epoch > 0 and initial_epoch < epochs:
                    print(f"⚠️ TIẾP TỤC: {region} từ epoch {initial_epoch}/{epochs}")
                    # Nạp lại mô hình từ checkpoint
                    continue_model = tf.keras.models.load_model(model_path)
                else:
                    # Nếu đã đủ epochs nhưng không có kết quả, huấn luyện lại
                    print(f"⚠️ PHÁT HIỆN: {region} đã hoàn thành {initial_epoch} epochs nhưng không có đủ kết quả. Huấn luyện lại...")
                    initial_epoch = 0
            except Exception as e:
                print(f"Lỗi khi nạp checkpoint: {str(e)}")
                initial_epoch = 0
        elif os.path.exists(model_path):
            print(f"⚠️ PHÁT HIỆN: {region} có file model nhưng không có thông tin checkpoint. Huấn luyện lại...")
        
        print(f"\n🔁 {'TIẾP TỤC' if initial_epoch > 0 else 'BẮT ĐẦU'} HUẤN LUYỆN CHO VÙNG: {region}")
        
        # Truyền initial_epoch và model (nếu có) vào hàm train_model
        model, history = train_model(
            base_dir, region, model_name,
            img_size=(224, 224),
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            continue_model=continue_model
        )
        
        if history:
            results[region] = {
                'accuracy': history.history.get('val_accuracy', [-1])[-1],
                'auc': history.history.get('val_auc', [-1])[-1]
            }

    # Tổng kết kết quả
    print("\n" + "="*60)
    print(f"TÓM TẮT KẾT QUẢ HUẤN LUYỆN MÔ HÌNH {model_name.upper()}")
    print("="*60)
    for region, metrics in results.items():
        acc = metrics['accuracy']
        auc_score = metrics['auc']
        print(f"{region}: Accuracy = {acc:.4f}, AUC = {auc_score:.4f}")
    print("="*60)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình phát hiện gãy xương')
    parser.add_argument('--base_dir', type=str, default='C:\\Users\\Admin\\Downloads\\fonai\\MURA\\mura_final',
                      help='Đường dẫn thư mục gốc chứa dữ liệu')
    parser.add_argument('--model', type=str, choices=['densenet121', 'resnet50v2', 'both'],
                      default='densenet121', help='Kiến trúc mô hình')
    parser.add_argument('--batch_size', type=int, default=8, help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=10, help='Số epoch')
    parser.add_argument('--img_size', type=int, default=224, help='Kích thước ảnh')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("THÔNG TIN HUẤN LUYỆN:")
    print(f"- Thư mục dữ liệu: {args.base_dir}")
    print(f"- Mô hình: {args.model}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Số epoch: {args.epochs}")
    print(f"- Kích thước ảnh: {args.img_size}x{args.img_size}")
    print("="*60 + "\n")
    
    # Kiểm tra xem các thư mục output có tồn tại không
    for output_dir in ['models', 'results']:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Đã tạo thư mục '{output_dir}'")
    
    # Huấn luyện dựa vào mô hình được chọn
    if args.model == 'both':
        print("=== HUẤN LUYỆN CẢ HAI MÔ HÌNH CHO TẤT CẢ CÁC VÙNG ===")
        
        print("\n=== HUẤN LUYỆN DENSENET121 CHO TẤT CẢ CÁC VÙNG ===")
        train_all_regions('densenet121', args.batch_size, args.epochs, args.base_dir)
        
        print("\n=== HUẤN LUYỆN RESNET50V2 CHO TẤT CẢ CÁC VÙNG ===")
        train_all_regions('resnet50v2', args.batch_size, args.epochs, args.base_dir)
    else:
        print(f"=== HUẤN LUYỆN {args.model.upper()} CHO TẤT CẢ CÁC VÙNG ===")
        train_all_regions(args.model, args.batch_size, args.epochs, args.base_dir)