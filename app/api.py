# app/api.py
from fastapi import FastAPI, UploadFile, File
from app.utils import detect_and_process_id_card
import numpy as np
import cv2

main = FastAPI()

@main.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Convert uploaded file to OpenCV image
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run your detection + OCR
    result = detect_and_process_id_card(image)

    return result
