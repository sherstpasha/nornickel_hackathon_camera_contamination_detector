import os
import random
from PIL import Image, ImageEnhance
import numpy as np
from joblib import Parallel, delayed

# ========== НАСТРОЙКИ ==========
INPUT_IMAGE_FOLDER = r"C:\Users\user\Desktop\nornik\orig_aug_data\open_img\train"
INPUT_MASK_FOLDER = r"C:\Users\user\Desktop\nornik\orig_aug_data\open_msk\train"
OUTPUT_FOLDER = r"C:\Users\user\Desktop\nornik\orig_aug_data\output"
NUM_IMAGES_TO_GENERATE = 500  # Количество изображений для создания
N_JOBS = 4  # Количество потоков для параллельной обработки

# Папки для вывода
OUTPUT_IMAGES_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
OUTPUT_MASKS_FOLDER = os.path.join(OUTPUT_FOLDER, "masks")

# Создание папок, если их нет
os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_MASKS_FOLDER, exist_ok=True)

# ========== ПОИСК СООТВЕТСТВУЮЩЕЙ МАСКИ ==========
def find_matching_mask(image_name, mask_folder):
    base_name, _ = os.path.splitext(image_name)
    possible_extensions = ['.png', '.jpg', '.jpeg']
    
    for ext in possible_extensions:
        mask_path = os.path.join(mask_folder, f"{base_name}{ext}")
        if os.path.exists(mask_path):
            return mask_path
    return None

# Создание пустой маски, если отсутствует
def create_missing_mask(mask_path, image_size):
    mask = Image.new("L", image_size, 0)  # Чёрная маска
    mask.save(mask_path)

# ========== АУГМЕНТАЦИЯ ==========
def random_augmentations(image, mask):
    if random.choice([True, False]):
        angle = random.choice([90, 180, 270])
        image = image.rotate(angle)
        mask = mask.rotate(angle)

    if random.choice([True, False]):
        flip_type = random.choice(["LR", "TB"])
        if flip_type == "LR":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    if random.choice([True, False]):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))

    if random.choice([True, False]):
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.7, 1.3))

    return image, mask

# ========== ОБРАБОТКА ==========
def process_single_image(output_counter, image_file):
    image_path = os.path.join(INPUT_IMAGE_FOLDER, image_file)
    mask_path = find_matching_mask(image_file, INPUT_MASK_FOLDER)

    if not mask_path:
        mask_path = os.path.join(INPUT_MASK_FOLDER, f"{os.path.splitext(image_file)[0]}.png")
        with Image.open(image_path) as img:
            create_missing_mask(mask_path, img.size)

    image = Image.open(image_path)
    mask = Image.open(mask_path).convert("L")

    # Аугментация
    aug_image, aug_mask = random_augmentations(image, mask)

    # Сохранение итоговых изображений и масок с одинаковыми именами
    file_name = f"{output_counter:04}.png"
    aug_image.save(os.path.join(OUTPUT_IMAGES_FOLDER, file_name))
    aug_mask.save(os.path.join(OUTPUT_MASKS_FOLDER, file_name))

def process_images_and_masks(num_images_to_generate, n_jobs):
    image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Нет изображений для обработки.")
        return

    # Генерация случайных индексов для изображений
    random_image_files = [random.choice(image_files) for _ in range(num_images_to_generate)]

    # Параллельная обработка
    Parallel(n_jobs=n_jobs)(
        delayed(process_single_image)(idx + 1, image_file)
        for idx, image_file in enumerate(random_image_files)
    )

# ========== ЗАПУСК ==========
process_images_and_masks(NUM_IMAGES_TO_GENERATE, N_JOBS)
print(f"Обработка завершена! Создано {NUM_IMAGES_TO_GENERATE} изображений.")
