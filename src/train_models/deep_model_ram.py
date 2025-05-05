import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import pickle
from tqdm import tqdm

# --- создаём папки для результатов ---
pth_dir = 'src/train_models/results/pth'
img_dir = 'src/train_models/results/img'
os.makedirs(pth_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# Определение датасета
class DurationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels.values if isinstance(labels, pd.Series) else labels).view(-1, 1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
koef_dropout = 0.5
# Определение модели
class DurationRegressor(nn.Module):
    def __init__(self, input_size):
        super(DurationRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),  # первый BatchNorm
            nn.Dropout(koef_dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),   # третий BatchNorm
            nn.Dropout(koef_dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),   # третий BatchNorm
            nn.Dropout(koef_dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),   # четвертый BatchNorm
            nn.Dropout(koef_dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),    # пятый BatchNorm
            nn.Dropout(koef_dropout),
            # nn.Linear(64, 32),
            # nn.LeakyReLU(0.1),
            # nn.BatchNorm1d(32),    # пятый BatchNorm
            # nn.Dropout(koef_dropout),
            nn.Linear(64, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

# Определение функции потерь MRE
class MRELoss(nn.Module):
    def __init__(self):
        super(MRELoss, self).__init__()
    
    def forward(self, pred, true):
        relative_error = torch.abs(true - pred) / (true)  # добавляем малое число для избежания деления на 0
        return torch.mean(relative_error)

# параметры для обучения
train_params = [
    'geom_surface_points', 
    'nodes', 
    'elements', 
    'mesh_surface_points'
]
# параметр для предсказания
# !!! текущая реализация заточена под один параметр для предсказания
# для предсказания нескольких параметров можно сделать копию этого файла и поменять параметр для предсказания
predict_parameters = ['ram', ]

def load_data(data_dir):
    data = []
    class_counts = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file == 'calc.json':
                with open(os.path.join(root, file), 'r') as f:
                    json_data = json.load(f)
                    if (json_data['ram'] // 1024) > 0: # берем ненулевые значение MB
                        data.append(
                            {param: json_data[param] if param != predict_parameters[0] else json_data[param] // 1024 for param in train_params + predict_parameters}
                        )
    return pd.DataFrame(data)

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

print("Загрузка данных...")
# !!! Тут указываем путь к папке с данынми
df = load_data('src/learn_data/')

X = df[train_params]
y = df[predict_parameters[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Анализ распределения данных
print("\nАнализ распределения данных:")
print(f"Train размер: {len(y_train)}, Test размер: {len(y_test)}")
print(f"Train среднее: {y_train.mean():.2f}, медиана: {y_train.median():.2f}, std: {y_train.std():.2f}")
print(f"Test среднее: {y_test.mean():.2f}, медиана: {y_test.median():.2f}, std: {y_test.std():.2f}")
print(f"Train мин/макс: {y_train.min():.2f}/{y_train.max():.2f}")
print(f"Test мин/макс: {y_test.min():.2f}/{y_test.max():.2f}")

# Анализ выбросов (значения больше 3 стандартных отклонений)
train_outliers = y_train[abs(y_train - y_train.mean()) > 3 * y_train.std()]
test_outliers = y_test[abs(y_test - y_test.mean()) > 3 * y_test.std()]
print(f"\nКоличество выбросов в Train: {len(train_outliers)}, {(len(train_outliers)/len(y_train)*100):.2f}%")
print(f"Количество выбросов в Test: {len(test_outliers)}, {(len(test_outliers)/len(y_test)*100):.2f}%")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import pickle
with open(os.path.join(pth_dir, 'scaler_ram_regression_deep_model_10s.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Использую X_train_scaled, y_train напрямую
train_dataset = DurationDataset(X_train_scaled, y_train)  # возвращаем к ram, а не классам
test_dataset = DurationDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\nРазмер обучающей выборки: {len(train_dataset)}")
print(f"Размер валидационной выборки: {len(test_dataset)}\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DurationRegressor(input_size=X_train.shape[1]).to(device)

criterion = nn.MSELoss()
# criterion = MRELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15, min_lr=0.00001)

print("Обучение модели...")
num_epochs = 100
patience = 10
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
train_maes = []
val_maes = []
train_mres = []
val_mres = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_preds = []
    train_targets = []
    
    for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_preds.extend(outputs.detach().cpu().numpy().flatten())
        train_targets.extend(batch_y.cpu().numpy().flatten())
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            val_preds.extend(outputs.detach().cpu().numpy().flatten())
            val_targets.extend(batch_y.cpu().numpy().flatten())
    val_loss /= len(test_loader)
    
    # Считаем MAE и MRE для обеих выборок
    train_mae = calculate_mae(train_targets, train_preds)
    val_mae = calculate_mae(val_targets, val_preds)
    
    train_mre = np.mean(np.abs(np.array(train_targets) - np.array(train_preds)) / np.array(train_targets)) * 100
    val_mre = np.mean(np.abs(np.array(val_targets) - np.array(val_preds)) / np.array(val_targets)) * 100
    
    # Сохраняем метрики (сохраняем для последующего анализа)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_maes.append(train_mae)
    val_maes.append(val_mae)
    train_mres.append(train_mre)
    val_mres.append(val_mre)
    
    if epoch % 10 == 0:
        print(f"\nЭпоха {epoch}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")
        print(f"Train MRE: {train_mre:.2f}%, Val MRE: {val_mre:.2f}%")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(pth_dir, 'best_ram_regression_deep_model.pth'))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break

# Сохраняем метрики для дальнейшего анализа
import pickle
with open(os.path.join(pth_dir, 'ram_train_losses.pkl'), 'wb') as f:
    pickle.dump(train_losses, f)
with open(os.path.join(pth_dir, 'ram_val_losses.pkl'), 'wb') as f:
    pickle.dump(val_losses, f)
with open(os.path.join(pth_dir, 'ram_train_maes.pkl'), 'wb') as f:
    pickle.dump(train_maes, f)
with open(os.path.join(pth_dir, 'ram_val_maes.pkl'), 'wb') as f:
    pickle.dump(val_maes, f)
with open(os.path.join(pth_dir, 'ram_train_mres.pkl'), 'wb') as f:
    pickle.dump(train_mres, f)
with open(os.path.join(pth_dir, 'ram_val_mres.pkl'), 'wb') as f:
    pickle.dump(val_mres, f)

# График MAE
new_train_maes = np.array(train_maes)
plt.figure(figsize=(10, 6))
plt.plot(new_train_maes, label='MAE обучения')
plt.plot(val_maes, label='MAE валидации')
plt.xlabel('Эпоха')
plt.ylabel('MAE')
plt.title('График MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'ram_mae_history.png'))
plt.savefig(os.path.join(img_dir, 'ram_mae_history.svg'))
plt.close()

# График MRE
new_train_mres = np.array(train_mres)
plt.figure(figsize=(10, 6))
plt.plot(new_train_mres, label='MRE обучения')
plt.plot(val_mres, label='MRE валидации')
plt.xlabel('Эпоха')
plt.ylabel('MRE, %')
plt.title('График MRE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'ram_mre_history.png'))
plt.savefig(os.path.join(img_dir, 'ram_mre_history.svg'))
plt.close()

print(f'Среднее значение ram в датасете: {df["ram"].mean():.2f}') 

model.eval()
test_preds = []
test_targets = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        test_preds.extend(outputs.cpu().numpy().flatten())
        test_targets.extend(batch_y.cpu().numpy().flatten())
plt.figure(figsize=(7,7))
plt.scatter(test_targets, test_preds, alpha=0.5)
plt.xlabel('Действительное значение времени расчета')
plt.ylabel('Предсказанное значение времени расчета')
plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'scatter_true_vs_pred_deep_model_ram.png')) 
plt.savefig(os.path.join(img_dir, 'scatter_true_vs_pred_deep_model_ram.svg'))