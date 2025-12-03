import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn

# --- КОНФІГУРАЦІЯ ---
# Переконайтеся, що шлях правильний
ROOT_DIR = r'C:\Users\User\Desktop\Dataset 1.0' 
CLASS_FOLDERS = ['Deserts', 'Forest Cover', 'Mountains']
IMG_SIZE = 480
NUM_CLASSES = len(CLASS_FOLDERS)
MODEL_PATH = 'best_terrain_model.pth'

class TerrainFolderDataset(Dataset):
    def __init__(self, root_split_dir, transform=None):
        self.transform = transform
        self.images_root = os.path.join(root_split_dir, 'images')
        self.samples = [] 

        if not os.path.exists(self.images_root):
             raise FileNotFoundError(f"Папка images не знайдена в {root_split_dir}")

        for class_idx, class_name in enumerate(CLASS_FOLDERS):
            class_dir = os.path.join(self.images_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    full_path = os.path.join(class_dir, fname)
                    # Зберігаємо шлях і ID класу
                    self.samples.append((full_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Помилка відкриття {img_path}: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        if self.transform:
            image = self.transform(image)
        
        mask_np = np.full((IMG_SIZE, IMG_SIZE), class_id, dtype=np.int64)
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
        
        return image, mask_tensor, img_path

def get_model(num_classes):

    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
    
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(10, num_classes, kernel_size=(1, 1))
    return model

def visualize_prediction(image_pil, true_mask, pred_mask, img_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_pil)
    plt.title(f"Image: {os.path.basename(img_path)}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap='viridis', vmin=0, vmax=NUM_CLASSES-1)
    plt.title(f"True: {CLASS_FOLDERS[true_mask[0,0]]}") 
    plt.axis('off')

    values, counts = np.unique(pred_mask, return_counts=True)
    if len(counts) > 0:
        majority_class_idx = values[np.argmax(counts)]
        pred_label = CLASS_FOLDERS[majority_class_idx]
    else:
        pred_label = "Unknown"
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap='viridis', vmin=0, vmax=NUM_CLASSES-1)
    plt.title(f"Pred: {pred_label}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Пристрій: {device}")

    print(f"Завантаження ваг з {MODEL_PATH}...")
    model = get_model(NUM_CLASSES).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✅ Модель успішно завантажена.")
        except RuntimeError as e:
            print(f"❌ Помилка ключів моделі: {e}")
            return
    else:
        print(f"❌ Файл {MODEL_PATH} не знайдено!")
        return
    
    model.eval()

    test_dir = os.path.join(ROOT_DIR, 'test')
    transforms_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        test_dataset = TerrainFolderDataset(test_dir, transform=transforms_val)
    except FileNotFoundError as e:
        print(e)
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Починаємо тестування на {len(test_dataset)} зображеннях...")
    
    total_pixels = 0
    correct_pixels = 0
    correct_images = 0
    
    visualize_count = 3  
    shown = 0

    with torch.no_grad():
        for i, (image, mask, img_path_tuple) in enumerate(test_loader):
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)['out']
            pred_mask = torch.argmax(output, dim=1) 

            correct_pixels += (pred_mask == mask).sum().item()
            total_pixels += mask.numel()

            pred_class = torch.mode(pred_mask.view(-1))[0].item()
            true_class = mask[0, 0, 0].item() 
            
            if pred_class == true_class:
                correct_images += 1

            if shown < visualize_count:
                mask_np = mask.cpu().squeeze().numpy()
                pred_np = pred_mask.cpu().squeeze().numpy()
                
                current_path = img_path_tuple[0]
                pil_img = Image.open(current_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                
                visualize_prediction(pil_img, mask_np, pred_np, current_path)
                shown += 1

    pixel_acc = 100 * correct_pixels / total_pixels if total_pixels > 0 else 0
    image_acc = 100 * correct_images / len(test_dataset) if len(test_dataset) > 0 else 0

    print("\n" + "="*30)
    print("       РЕЗУЛЬТАТИ ТЕСТУ")
    print("="*30)
    print(f"Pixel Accuracy (Точність пікселів):   {pixel_acc:.2f}%")
    print(f"Image Accuracy (Точність класів):     {image_acc:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()