import os
import sys
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import differential_evolution

# Пути к данным и модели
model_path = r"C:\Users\pasha\OneDrive\Рабочий стол\best.pt"
dataset_path = (
    r"C:\Users\pasha\OneDrive\Рабочий стол\train_dataset\cv_open_dataset\open_img"
)
gt_path = r"C:\Users\pasha\OneDrive\Рабочий стол\train_dataset\cv_open_dataset\open_msk"

# Переводим модель на GPU
model = YOLO(model_path, verbose=False)
model.to(0)  # Используем GPU

all_images = [f for f in os.listdir(dataset_path) if f.lower().endswith(".jpg")]


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


def convert_mask_to_yolo(mask_image):
    if mask_image is None:
        return None
    black_mask = cv2.inRange(mask_image, (0, 0, 0), (50, 50, 50))
    new_image = np.ones_like(mask_image) * 255
    new_image[black_mask > 0] = [0, 0, 0]
    return new_image


def iou(y_true, y_pred, class_label):
    y_true = y_true == class_label
    y_pred = y_pred == class_label
    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 1.0
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    return inter / (union + 1e-8)


def create_mask_from_results(image, results, contour_scale, epsilon_factor):
    height, width = image.shape[:2]
    final_mask = np.zeros((int(height), int(width)), dtype=np.uint8)

    masks = results.masks
    if masks is not None:
        for mask_array in masks.data:
            # Переводим тензор на CPU перед вызовом numpy()
            mask_i = mask_array.cpu().numpy()
            if mask_i.ndim == 2:
                mask_i_resized = cv2.resize(
                    mask_i, (width, height), interpolation=cv2.INTER_LINEAR
                )
            else:
                continue

            bin_mask = (mask_i_resized > 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx = x + w / 2
                    cy = y + h / 2

                contour = contour.astype(np.float32)
                contour[:, 0, 0] -= cx
                contour[:, 0, 1] -= cy
                contour = contour * contour_scale
                contour[:, 0, 0] += cx
                contour[:, 0, 1] += cy

                eps = epsilon_factor * cv2.arcLength(contour, True)
                smoothed_contour = cv2.approxPolyDP(contour, eps, True)
                cv2.drawContours(
                    final_mask,
                    [smoothed_contour.astype(int)],
                    -1,
                    255,
                    thickness=cv2.FILLED,
                )

    return final_mask


def evaluate_params(params):
    brightness_factor, contrast_factor, noise_factor, contour_scale, epsilon_factor = (
        params
    )

    iou_scores = []
    for image_name in all_images:
        image_path = os.path.join(dataset_path, image_name)
        gt_mask_name = image_name.replace(".jpg", ".png")
        gt_mask_path = os.path.join(gt_path, gt_mask_name)
        if not os.path.exists(gt_mask_path):
            continue

        orig_image = cv2.imread(image_path)
        if orig_image is None:
            continue

        processed_image = preprocess_image(
            orig_image, brightness_factor, contrast_factor, noise_factor
        )

        results_list = model.predict(
            processed_image,
            conf=0.21,
            verbose=False,
            device=0,
            workers=5,
        )
        if len(results_list) == 0:
            continue
        results = results_list[0]

        pred_mask = create_mask_from_results(
            orig_image, results, contour_scale, epsilon_factor
        )

        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_COLOR)
        if gt_mask is None:
            continue

        gt_mask_yolo = convert_mask_to_yolo(gt_mask)
        pred_mask_yolo = convert_mask_to_yolo(
            cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
        )

        if gt_mask_yolo is not None and pred_mask_yolo is not None:
            score = (
                iou(gt_mask_yolo, pred_mask_yolo, 255)
                + iou(gt_mask_yolo, pred_mask_yolo, 0)
            ) / 2
            iou_scores.append(score)

    if len(iou_scores) > 0:
        mean_iou = np.mean(iou_scores)
    else:
        mean_iou = 0.0

    print(-mean_iou)
    # Убираем print отсюда, чтобы не мешать parallel execution
    return -mean_iou


if __name__ == "__main__":
    # Диапазоны параметров
    bounds = [
        (1.0, 1.5),  # brightness_factor
        (1.0, 1.5),  # contrast_factor
        (0.0, 0.5),  # noise_factor
        (1.0, 1.3),  # contour_scale
        (0.0, 0.03),  # epsilon_factor
    ]

    result = differential_evolution(
        evaluate_params,
        bounds,
        maxiter=10,
        popsize=5,
        tol=0.01,
        disp=True,
        workers=5,
        updating="deferred",
    )
    best_params = result.x
    best_score = -result.fun

    print("Лучшие параметры:", best_params)
    print("Достигнутый mean IoU:", best_score)
