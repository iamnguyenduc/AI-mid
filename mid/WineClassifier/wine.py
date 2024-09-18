import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Đường dẫn tới file dữ liệu
data_file_path = "wine.data"
names_file_path = "wine.names"

# Bước 1: Đọc dữ liệu từ file wine.data
# Đặt tên cột cho dữ liệu
column_names = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", 
                "Magnesium", "Total phenols", "Flavanoids", 
                "Nonflavanoid phenols", "Proanthocyanins", 
                "Color intensity", "Hue", "OD280/OD315 of diluted wines", 
                "Proline"]

# Đọc dữ liệu từ file wine.data
df = pd.read_csv(data_file_path, header=None, names=column_names)

# Hiển thị 5 dòng đầu tiên để kiểm tra
# print("Dữ liệu mẫu từ file wine.data:")
# print(df.head())

# Bước 2: Đọc mô tả từ file wine.names
with open(names_file_path, "r") as f:
    names_content = f.read()

# In nội dung của file wine.names (mô tả)
# print("\nNội dung file wine.names:")
# print(names_content)

# Bước 3: Chọn mẫu test và tập train
# Đặt chỉ số của mẫu test (thay đổi giá trị này để chọn mẫu khác)
test_index = 4  # Ví dụ: Chọn mẫu thứ 5 (index 4)

# Chọn mẫu test
test_sample = df.iloc[test_index, 1:].values  # Lấy toàn bộ đặc trưng của mẫu test (trừ cột 'Class')
test_label = df.iloc[test_index, 0]  # Lấy nhãn (Class) của mẫu test

# Xóa mẫu test khỏi tập train
df_train = df.drop(index=test_index)
X_train = df_train.iloc[:, 1:].values  # Tập train (đặc trưng)
y_train = df_train.iloc[:, 0].values   # Nhãn của tập train

# Bước 4: Tính khoảng cách Euler từ mẫu test đến tất cả mẫu train
distances = euclidean_distances([test_sample], X_train)[0]

# Tìm mẫu train gần nhất (khoảng cách bé nhất)
nearest_index = np.argmin(distances)
predicted_label = y_train[nearest_index]

# Giá trị dmin - khoảng cách bé nhất
dmin = distances[nearest_index]

# In ra thông tin chi tiết về mẫu train gần nhất
nearest_train_sample = df_train.iloc[nearest_index, 1:]  # Mẫu train gần nhất
nearest_train_class = df_train.iloc[nearest_index, 0]    # Class của mẫu train gần nhất

# Bước 5: In ra thông tin chi tiết
print(f"Mẫu test: {test_index + 1}")
print(f"Khoảng cách nhỏ nhất (dmin): {dmin}")
print(f"Mẫu train gần nhất là mẫu thứ {nearest_index + 1}, thuộc Class {nearest_train_class}.")
#print(f"Giá trị của mẫu train gần nhất:\n{nearest_train_sample}")
print(f"Label dự đoán: {predicted_label}, Label thực tế: {test_label}")

# Bước 6: Kiểm tra kết quả
if predicted_label == test_label:
    print("Phân loại chính xác!")
else:
    print(f"Phân loại sai! Mẫu test thuộc Class {test_label} nhưng được dự đoán là Class {predicted_label}.")
