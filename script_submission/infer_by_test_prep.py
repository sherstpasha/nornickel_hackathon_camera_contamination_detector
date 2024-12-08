import os
import sys
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO

# Программа получает на вход два аргумента: путь к датасету и путь к выходному файлу
dataset_path, output_path = sys.argv[1:]

# Оптимизированные параметры (подставьте найденные значения)
best_brightness = 1.2  # Пример значения
best_contrast = 1.3  # Пример значения
best_noise = 0.4  # Пример значения
best_contour_scale = 1.15  # Пример значения
best_epsilon = 0.02  # Пример значения


# Функция предобработки изображения
def preprocess_image(image, brightness_factor, contrast_factor, noise_factor):
    img = image.astype(np.float32)
    mean_val = np.mean(img)
    img = img * brightness_factor
    img = mean_val + (img - mean_val) * contrast_factor
    img = np.clip(img, 0, 255).astype(np.uint8)

    if noise_factor > 0:
        sigma_c = 75 * noise_factor
        sigma_s = 75 * noise_factor
        img = cv2.bilateralFilter(img, d=9, sigmaColor=sigma_c, sigmaSpace=sigma_s)
    return img


# Функция инференса модели
def infer_image(model, image_path):
    image = cv2.imread(image_path)
    # Применяем предобработку с лучшими параметрами
    processed_image = preprocess_image(
        image, best_brightness, best_contrast, best_noise
    )
    return model(processed_image, conf=0.5)


# Укажите корректный путь до вашей модели, которая будет находиться в проекте
model_path = "./baseline.pt"
example_model = YOLO(model_path)
example_model.to("cpu")  # Используем CPU или замените на '0' для GPU


# Функция создания маски с использованием оптимизированных параметров
def create_mask(image_path, results):
    # Загружаем изображение и определяем размеры
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Итоговая маска
    final_mask = np.zeros((height, width), dtype=np.uint8)

    # Проходим по результатам
    for result in results:
        masks = result.masks
        if masks is not None:
            for mask_array in masks.data:
                mask_i = (
                    mask_array.cpu().numpy()
                )  # Переводим на CPU перед использованием

                # Изменяем размер маски под изображение
                mask_i_resized = cv2.resize(
                    mask_i, (width, height), interpolation=cv2.INTER_LINEAR
                )
                bin_mask = (mask_i_resized > 0).astype(np.uint8)

                # Находим контуры в бинарной маске
                contours, _ = cv2.findContours(
                    bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    # Находим центр контура для масштабирования
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                    else:
                        x, y, w, h = cv2.boundingRect(contour)
                        cx = x + w / 2
                        cy = y + h / 2

                    # Переносим контур к центру, масштабируем и возвращаем обратно
                    contour = contour.astype(np.float32)
                    contour[:, 0, 0] -= cx
                    contour[:, 0, 1] -= cy
                    contour = contour * best_contour_scale  # Используем лучший масштаб
                    contour[:, 0, 0] += cx
                    contour[:, 0, 1] += cy

                    # Сглаживаем контур
                    epsilon = best_epsilon * cv2.arcLength(contour, True)
                    smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)

                    # Рисуем сглаженные и увеличенные контуры на итоговой маске
                    cv2.drawContours(
                        final_mask,
                        [smoothed_contour.astype(int)],
                        -1,
                        255,
                        thickness=cv2.FILLED,
                    )

    return final_mask


# Словарь для накопления результатов
results_dict = {}

# Производим инференс для каждого изображения и сохраняем маски в один JSON файл
for image_name in os.listdir(dataset_path):
    if image_name.lower().endswith(".jpg"):
        results = infer_image(example_model, os.path.join(dataset_path, image_name))
        mask = create_mask(os.path.join(dataset_path, image_name), results)

        # Кодируем маску в PNG в память
        _, encoded_img = cv2.imencode(".png", mask)
        # Кодируем в base64, чтобы поместить в JSON
        encoded_str = base64.b64encode(encoded_img).decode("utf-8")
        results_dict[image_name] = encoded_str

# Сохраняем результаты в один файл "submit" (формат JSON)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, ensure_ascii=False)
