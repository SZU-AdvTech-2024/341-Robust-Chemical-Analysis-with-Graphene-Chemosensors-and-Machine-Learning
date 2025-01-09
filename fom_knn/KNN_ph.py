import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import combinations
import torch
from torch.utils.data import DataLoader
import time

# 记录开始时间
start_time = time.time()

# Load all data from the directory and preprocess
data_dir = "C:\\Users\\30300\\Desktop\\learn_code\\Chemosensing-main\\AnalyteFOM_pH Buffer"

# Preload all CSV data into a dictionary
def load_data(data_dir):
    X = []
    y = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            # 提取类别信息
            class_label = int(file_name.split("_")[2].split(' ')[1])
            # 读取文件
            file_path = os.path.join(data_dir, file_name)
            data = pd.read_csv(file_path, header=None).values.flatten()
            X.append(data)
            y.append(class_label)
    return np.array(X), np.array(y)

# Load data
data, labels = load_data(data_dir)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)

# Move data to GPU
def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32).cuda(), torch.tensor(y, dtype=torch.long).cuda()

X_train, y_train = to_tensor(X_train, y_train)
X_test, y_test = to_tensor(X_test, y_test)

# GPU-accelerated KNN function
def gpu_knn(X_train, y_train, X_test, k=30):
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]

    # Expand dimensions for broadcasting
    X_train_exp = X_train.unsqueeze(0).repeat(num_test, 1, 1)  # Shape: (num_test, num_train, num_features)
    X_test_exp = X_test.unsqueeze(1).repeat(1, num_train, 1)   # Shape: (num_test, num_train, num_features)

    # Compute L2 distances
    distances = torch.sqrt(torch.sum((X_train_exp - X_test_exp) ** 2, dim=2))  # Shape: (num_test, num_train)

    # Sort distances and get top-k indices
    sorted_indices = torch.argsort(distances, dim=1)[:, :k]

    # Vote for classes
    top_k_labels = y_train[sorted_indices]  # Shape: (num_test, k)
    predictions = torch.mode(top_k_labels, dim=1)[0]  # Shape: (num_test,)

    return predictions

# Perform feature combination and KNN classification
results = []
for num_features in range(1, 21):  # From 1 to 20 features
    feature_combinations = list(combinations(range(data.shape[1]), num_features))

    for feature_set in feature_combinations:
        selected_features_train = X_train[:, feature_set]
        selected_features_test = X_test[:, feature_set]

        # Predict using GPU-accelerated KNN
        predictions = gpu_knn(selected_features_train, y_train, selected_features_test, k=5)

        # Calculate accuracy
        accuracy = (predictions == y_test).float().mean().item()
        results.append((num_features, feature_set, accuracy))

        # Print progress
        print(f"Features: {num_features}, Accuracy: {accuracy:.2%}")
# 记录结束时间
end_time = time.time()

# 计算总用时
execution_time = end_time - start_time

# 输出总用时
print(f"总用时：{execution_time}秒")

# Save results to CSV
results_df = pd.DataFrame(results, columns=["Num_Features", "Feature_Set", "Accuracy"])
results_df.to_csv("knn_results.csv", index=False)
