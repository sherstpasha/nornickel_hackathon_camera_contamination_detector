import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(input_dir, output_dir, test_size=0.2):
    # Получаем список файлов
    files = os.listdir(input_dir)
    files = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]

    # Разделяем на обучение и тест
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

    # Создаем папки для train и test
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Копируем файлы
    for file in train_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(train_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(test_dir, file))

    print(f"Разделение данных завершено. Обучающие данные: {len(train_files)}, тестовые данные: {len(test_files)}")

# Указание путей
orig_img_path = r"C:\Users\user\Desktop\nornik\orig_data\open_img"
orig_msk_path = r"C:\Users\user\Desktop\nornik\orig_data\open_msk"

# Новые папки для разделенных данных
output_img_path = r"C:\Users\user\Desktop\nornik\split_data\open_img"
output_msk_path = r"C:\Users\user\Desktop\nornik\split_data\open_msk"

# Разделяем данные для open_img
split_data(orig_img_path, output_img_path)

# Разделяем данные для open_msk
split_data(orig_msk_path, output_msk_path)
