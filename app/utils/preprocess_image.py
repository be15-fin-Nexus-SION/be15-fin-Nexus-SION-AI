import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> Image.Image:
    gray_image = image.convert('L')

    gray = np.array(gray_image)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return Image.fromarray(thresh)
