import os
import sys
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# model_path - путь к вашей модели (с именем и расширением файла, относительно скрипта в вашем архиве проекта)
# dataset_path - путь к папке с тестовым датасетом.
# Он состоит из n фотографий c расширением .jpg (гарантируется, что будет только это расширение)
#
# output_path - путь к файлу, в который будут сохраняться результаты (с именем и расширением файла)
dataset_path, output_path = sys.argv[1:]


# TODO ваша работа с моделью
# на вход модели подаются изображения из тестовой выборки
# результатом должен стать json-файл
# В качестве примера здесь показана работа на примере модели из baseline

# Пример функции инференса модели
def infer_image(model, image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Инференс
    return model(image)


# TODO Ваш проект будет перенесен целиком, укажите корректны относительный путь до модели!!!
# TODO Помните, что доступа к интернету не будет и нельзя будет скачать веса модели откуда-то с внешнего ресурса!
model_path = './baseline.pt'

# Тут показан пример с использованием модели, полученной из бейзлайна
example_model = YOLO(model_path)
example_model.to('cpu')


def create_mask(image_path, results):
    # Загружаем изображение и переводим в градации серого
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Создаем пустую маску с черным фоном
    mask = np.zeros((height, width), dtype=np.uint8)

    # Проходим по результатам и создаем маску
    for result in results:
        masks = result.masks  # Получаем маски из результатов
        if masks is not None:
            for mask_array in masks.data:  # Получаем маски как массивы
                mask_i = mask_array.numpy()  # Преобразуем маску в numpy массив

                # Изменяем размер маски под размер оригинального изображения
                mask_i_resized = cv2.resize(mask_i, (width, height), interpolation=cv2.INTER_LINEAR)

                # Накладываем маску на пустую маску (255 для белого)
                mask[mask_i_resized > 0] = 255

    return mask

# Ваша задача - произвести инференс и сохранить маски НЕ в отдельные файлы, а в один файл submit.
# Для этого мы сначала будем накапливать результаты в словаре, а затем сохраним их в JSON.
results_dict = {}

for image_name in os.listdir(dataset_path):
    if image_name.lower().endswith(".jpg"):
        results = infer_image(example_model, os.path.join(dataset_path, image_name))
        mask = create_mask(os.path.join(dataset_path, image_name), results)
        
        # Кодируем маску в PNG в память
        _, encoded_img = cv2.imencode(".png", mask)
        # Кодируем в base64, чтобы поместить в JSON
        encoded_str = base64.b64encode(encoded_img).decode('utf-8')
        results_dict[image_name] = encoded_str

# Сохраняем результаты в один файл "submit" (формат JSON)
submit_path = os.path.join(output_path, "submit.json")
with open(submit_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, ensure_ascii=False)
