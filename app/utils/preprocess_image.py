import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> Image.Image:
    # PIL.Image → numpy array 변환
    img = np.array(image)

    # RGB → GRAY (흑백 변환)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 이진화 처리 (배경/글자 대비 강화)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # numpy array → PIL.Image 복원
    return Image.fromarray(thresh)
