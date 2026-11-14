from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    image_path = Path(__file__).with_name("kp.jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {image_path}")

    with Image.open(image_path) as image:
        original = np.array(image.convert("RGB"))

    height, width = original.shape[:2]

    scaled_half = cv2.resize(
        original,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_LINEAR,
    )
    scaled_double = cv2.resize(
        original,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_LINEAR,
    )

    center = (width // 2, height // 2)
    rotated_images = []
    for angle in (45, 90, 180):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(original, rotation_matrix, (width, height))
        rotated_images.append((f"Hasil Rotasi {angle} Derajat", rotated))

    translation_matrix = np.float32([[1, 0, 150], [0, 1, 75]])
    translated = cv2.warpAffine(
        original,
        translation_matrix,
        (width, height),
    )

    images = [
        ("Citra Asli", original),
        ("Hasil Translasi (150, 75)", translated),
        ("Hasil Scaling 0,5x", scaled_half),
        ("Hasil Scaling 2x", scaled_double),
        *rotated_images,
        ("Horizontal Flip", cv2.flip(original, 1)),
        ("Vertical Flip", cv2.flip(original, 0)),
    ]

    figure, axes = plt.subplots(3, 3, figsize=(15, 12))
    for axis, (title, image) in zip(axes.flat, images):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    plt.show()
