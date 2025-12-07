from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

ocr = PaddleOCR(lang='ar', use_angle_cls=False)
result = ocr.ocr("test_ids/mohamed_id.jpg")

image = Image.open("test_ids/mohamed_id.jpg").convert('RGB')
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # or specify an Arabic‑supporting font

for line in result:
    for word in line:
        print(word)

image.save("paddle_annotated.jpg")
