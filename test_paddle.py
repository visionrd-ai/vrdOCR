from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import arabic_reshaper
# from bidi.algorithm import get_display

# Paths
pdf_path = '/home/amur/Amur/ForgeryDetectionV1.3/final_demo_files/original/1.png'
output_dir = 'annotated_pages_test'
os.makedirs(output_dir, exist_ok=True)

# Convert PDF to images
# pages = convert_from_path(pdf_path)
# Initialize Arabic OCR
ocr = PaddleOCR(lang='en', use_angle_cls=True)

# Load Arabic font (make sure this font file exists on your system)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Replace if needed
font = ImageFont.truetype(font_path, size=30)

# Drawing function using PIL
def draw_ocr_pil(pil_img, ocr_result):
    draw = ImageDraw.Draw(pil_img)

    for line in ocr_result:
        for box, (text, score) in line:
            box = np.array(box).astype(np.int32)

            # Draw polygon
            draw.line(list(box.reshape(-1)) + list(box[0]), fill="green", width=3)

            # Reshape + bidi Arabic text
            reshaped_text = arabic_reshaper.reshape(text)
            # bidi_text = get_display(reshaped_text)

            # Draw text
            x, y = box[0]
            draw.text((x, y - 25), reshaped_text, fill="blue", font=font)#bidi_text

    return pil_img

# # Process each page
# for idx, page in enumerate(pages):
#     img_cv2 = np.array(page)
#     result = ocr.ocr(img_cv2, cls=True)
#     img_annotated = draw_ocr_pil(page, result)

#     save_path = os.path.join(output_dir, f"page_{idx+1}_annotated.jpg")
#     img_annotated.save(save_path)
#     print(f"Saved: {save_path}")

page = cv2.imread(pdf_path)
page = Image.fromarray(page)
img_cv2 = np.array(page)
result = ocr.ocr(img_cv2, cls=True)
img_annotated = draw_ocr_pil(page, result)

save_path = os.path.join(output_dir, f"page_{idx+1}_annotated.jpg")
img_annotated.save(save_path)
print(f"Saved: {save_path}")
