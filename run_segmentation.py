import torch
import os
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import time

ROOT_DIR = r'C:\Users\User\Desktop\Dataset 1.0' 
CLASS_FOLDERS = ['Deserts', 'Forest Cover', 'Mountains'] 
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 20
IMG_SIZE = 480
NUM_CLASSES = len(CLASS_FOLDERS)

class TerrainFolderDataset(Dataset):
    def __init__(self, root_split_dir, transform=None):
        self.transform = transform
        self.images_root = os.path.join(root_split_dir, 'images')
        
        if not os.path.exists(self.images_root):
             raise FileNotFoundError(f"Папка images не знайдена в {root_split_dir}")

        self.samples = [] 

        for class_idx, class_name in enumerate(CLASS_FOLDERS):
            class_dir = os.path.join(self.images_root, class_name)
            
            if not os.path.isdir(class_dir):
                print(f"УВАГА: Папка класу '{class_name}' не знайдена в {self.images_root}. Пропускаємо.")
                continue
            
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    full_path = os.path.join(class_dir, fname)
                    self.samples.append((full_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Помилка відкриття файлу {img_path}: {e}")

            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)
        
        mask_np = np.full((IMG_SIZE, IMG_SIZE), class_id, dtype=np.int64)
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
        
        return image, mask_tensor

def get_model(num_classes):
    print(f"Завантаження DeepLabV3+ (MobileNetV3) для {num_classes} класів...")
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(10, num_classes, kernel_size=(1, 1))
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in loader:
       
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(images)
 
        loss = criterion(outputs['out'], masks) + 0.5 * criterion(outputs['aux'], masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs['out'], masks)
            running_loss += loss.item()
    return running_loss / len(loader)

def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Знайдено GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("⚠️ GPU не знайдено. Навчання буде повільним на CPU.")

    transforms_common = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(ROOT_DIR, 'train')
    val_dir = os.path.join(ROOT_DIR, 'validation') 
    test_dir = os.path.join(ROOT_DIR, 'test')

    print(f"\nШукаємо дані в: {ROOT_DIR}")
    
    try:
        train_dataset = TerrainFolderDataset(train_dir, transform=transforms_common)
        val_dataset = TerrainFolderDataset(val_dir, transform=transforms_common)
        test_dataset = TerrainFolderDataset(test_dir, transform=transforms_common)
    except FileNotFoundError as e:
        print(f"\nКРИТИЧНА ПОМИЛКА: {e}")
        return

    print(f"Знайдено зображень -> Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("Помилка: Тренувальний датасет порожній.")
        return
    num_workers = 2 if os.name == 'nt' else 4 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())

    model = get_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    start_time = time.time()

    print("\n--- Початок навчання ---")
    for epoch in range(EPOCHS):
        ep_start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        ep_duration = time.time() - ep_start
        print(f"Epoch {epoch+1}/{EPOCHS} | Time: {ep_duration:.0f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_terrain_model.pth')
            print("  --> ⭐ Збережено кращу модель")

    total_time = (time.time() - start_time) / 60
    print(f"\nЗагальний час навчання: {total_time:.1f} хв.")

    print("\n--- Фінальний тест ---")
    model.load_state_dict(torch.load('best_terrain_model.pth'))
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()