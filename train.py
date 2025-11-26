import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import time
import os

# [cite: 1475] Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# [cite: 1417] Định nghĩa Transforms
# Lưu ý: Báo cáo dùng RandomResizedCrop và RandomHorizontalFlip cho Train
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((75, 75)), # [cite: 1419]
        transforms.RandomResizedCrop(75),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # [cite: 1424]
    ]),
    'val': transforms.Compose([
        transforms.Resize((75, 75)), # [cite: 1428]
        transforms.CenterCrop(75),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_data(data_dir):
    # Load toàn bộ dataset từ thư mục ảnh
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    
    # [cite: 1448] Chia tập train/test tỉ lệ 8:2
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Gán transform riêng cho tập test (val) để không augmentation
    test_dataset.dataset.transform = data_transforms['val']
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2),
        'val': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}
    return dataloaders, dataset_sizes

def build_model():
    # [cite: 1477] Load ResNet18 pretrained
    model = models.resnet18(pretrained=True)
    
    # [cite: 1477] Thay thế lớp FC cuối cùng cho bài toán 2 lớp
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) 
    
    model = model.to(device)
    return model

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=150):
    start_time = time.time() # [cite: 1484]
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Mỗi epoch có pha training và validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # [cite: 1491]
            else:
                model.eval()   # [cite: 1520]

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # [cite: 1501]

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # [cite: 1505]
                    _, preds = torch.max(outputs, 1) # [cite: 1506]
                    loss = criterion(outputs, labels) # [cite: 1507]

                    # Backward + Optimize chỉ ở pha train
                    if phase == 'train':
                        loss.backward() # [cite: 1509]
                        optimizer.step() # [cite: 1510]

                # Thống kê
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}%') # [cite: 1518]

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model

if __name__ == "__main__":
    DATA_DIR = 'Images_Data' # Thư mục chứa ảnh từ bước preprocess
    
    if not os.path.exists(DATA_DIR):
        print("Không tìm thấy dữ liệu. Hãy chạy preprocess.py trước.")
    else:
        dataloaders, dataset_sizes = load_data(DATA_DIR)
        model = build_model()
        
        # [cite: 1479] Hàm mất mát CrossEntropy
        criterion = nn.CrossEntropyLoss()
        
        # [cite: 1480] Optimizer SGD, lr=0.001, momentum=0.9
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # [cite: 1483] Số epoch là 150
        model_ft = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=150)