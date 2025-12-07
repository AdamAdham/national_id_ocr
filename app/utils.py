# utils.py
import json
from ultralytics import YOLO
import cv2
import re
import easyocr
import pytesseract

CUSTOM_TESS_CONFIG = "-l ara --oem 3 --psm 7 tessedit_char_blacklist=0123456789"

# Initialize EasyOCR reader (this should be done once for efficiency)
YOLO_PATH = "artifacts/YOLO_models/"
reader = easyocr.Reader(["ar"], gpu=False)
yolo_id = YOLO(YOLO_PATH + "detect_id.pt")
yolo_fields = YOLO(YOLO_PATH + "detect_odjects.pt")
id_card_model = YOLO(YOLO_PATH + "detect_id_card.pt")


# Function to preprocess the cropped image
def preprocess_image(cropped_image):
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    return gray_image


# Functions for specific fields with custom OCR configurations
def extract_text(image, bbox, lang="ara"):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    preprocessed_image = preprocess_image(cropped_image)
    results = reader.readtext(preprocessed_image, detail=0, paragraph=True)
    text = " ".join(results)
    return text.strip()


def preprocess_for_tesseract(img):
    """Preprocess crop for small Arabic text: grayscale, upscale, denoise, binarize."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # upscale to give Tesseract more pixels to work with (helps short words)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # adaptive threshold or Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def extract_first_name(image, bbox):
    """
    image: full cropped ID card image (BGR)
    bbox: [x1, y1, x2, y2]
    returns: stripped string (Arabic)
    """
    x1, y1, x2, y2 = bbox
    # Small safety padding around the box
    pad_x = int((x2 - x1) * 0.08)
    pad_y = int((y2 - y1) * 0.12)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(image.shape[1], x2 + pad_x)
    y2p = min(image.shape[0], y2 + pad_y)

    crop = image[y1p:y2p, x1p:x2p]
    proc = preprocess_for_tesseract(crop)

    # Run Tesseract
    try:
        text = pytesseract.image_to_string(proc, config=CUSTOM_TESS_CONFIG)
    except Exception as e:
        # graceful fallback to EasyOCR if Tesseract crashes for some reason
        print("Tesseract error:", e)
        # fallback: use your existing EasyOCR reader (reader.readtext)
        results = reader.readtext(proc, detail=0, paragraph=False)
        text = " ".join(results)

    # Postprocess: remove digits and weird symbols, then strip
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)  # keep Arabic letters + spaces
    return text.strip()


# Function to detect national ID numbers in a cropped image
def detect_national_id(cropped_image):
    # model = YOLO("detect_id.pt")  # Load the model directly in the function
    results = yolo_id(cropped_image)
    detected_info = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_info.append((cls, x1))
            cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                cropped_image,
                str(cls),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

    detected_info.sort(key=lambda x: x[1])
    id_number = "".join([str(cls) for cls, _ in detected_info])

    return id_number


# Function to remove numbers from a string
def remove_numbers(text):
    return re.sub(r"\d+", "", text)


# Function to expand bounding box height only
def expand_bbox_height(bbox, scale=1.2, image_shape=None):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_height = int(height * scale)
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, image_shape[0])
    return [x1, new_y1, x2, new_y2]


# Function to process the cropped image
def process_image(cropped_image):
    # Load the trained YOLO model for objects (fields) detection
    # model = YOLO("detect_odjects.pt")
    results = yolo_fields(cropped_image)

    print("Detection Results:", results)

    # Variables to store extracted values
    first_name = ""
    second_name = ""
    merged_name = ""
    nid = ""
    address = ""
    serial = ""

    # Loop through the results
    print("results length:", len(results))
    for result in results:
        # output_path = "adam_annotated.jpg"
        # result.save(output_path)

        print("length of boxes:", len(result.boxes))
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            bbox = [int(coord) for coord in bbox]
            
            print(class_name, "confidence:", box.conf, "bbox:", bbox)

            if class_name == "firstName":
                first_name = extract_text(cropped_image, bbox, lang="ara")
                # first_name = extract_first_name(cropped_image, bbox)
            elif class_name == "lastName":
                second_name = extract_text(cropped_image, bbox, lang="ara")
            elif class_name == "serial":
                serial = extract_text(cropped_image, bbox, lang="eng")
            elif class_name == "address":
                address = extract_text(cropped_image, bbox, lang="ara")
            elif class_name == "nid":
                expanded_bbox = expand_bbox_height(
                    bbox, scale=1.5, image_shape=cropped_image.shape
                )
                cropped_nid = cropped_image[
                    expanded_bbox[1] : expanded_bbox[3],
                    expanded_bbox[0] : expanded_bbox[2],
                ]
                nid = detect_national_id(cropped_nid)

    merged_name = f"{first_name} {second_name}"
    print(f"First Name: {first_name}")
    print(f"Second Name: {second_name}")
    print(f"Full Name: {merged_name}")
    print(f"National ID: {nid}")
    print(f"Address: {address}")
    print(f"Serial: {serial}")

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump({"first Name": first_name, "second Name": second_name, "full_name": merged_name, "national_id": nid, "address": address, "serial": serial}, f, ensure_ascii=False, indent=4)

    decoded_info = decode_egyptian_id(nid)
    return (
        first_name,
        second_name,
        merged_name,
        nid,
        address,
        decoded_info["Birth Date"],
        decoded_info["Governorate"],
        decoded_info["Gender"],
    )


# Function to decode the Egyptian ID number
def decode_egyptian_id(id_number):
    governorates = {
        "01": "Cairo",
        "02": "Alexandria",
        "03": "Port Said",
        "04": "Suez",
        "11": "Damietta",
        "12": "Dakahlia",
        "13": "Ash Sharqia",
        "14": "Kaliobeya",
        "15": "Kafr El - Sheikh",
        "16": "Gharbia",
        "17": "Monoufia",
        "18": "El Beheira",
        "19": "Ismailia",
        "21": "Giza",
        "22": "Beni Suef",
        "23": "Fayoum",
        "24": "El Menia",
        "25": "Assiut",
        "26": "Sohag",
        "27": "Qena",
        "28": "Aswan",
        "29": "Luxor",
        "31": "Red Sea",
        "32": "New Valley",
        "33": "Matrouh",
        "34": "North Sinai",
        "35": "South Sinai",
        "88": "Foreign",
    }

    # if not id_number:
    #     return {"Birth Date": None, "Governorate": None, "Gender": None}
    
    century_digit = int(id_number[0])
    year = int(id_number[1:3])
    month = int(id_number[3:5])
    day = int(id_number[5:7])
    governorate_code = id_number[7:9]
    gender_code = int(id_number[12:13])

    if century_digit == 2:
        century = "1900-1999"
        full_year = 1900 + year
    elif century_digit == 3:
        century = "2000-2099"
        full_year = 2000 + year
    else:
        raise ValueError("Invalid century digit")

    gender = "Male" if gender_code % 2 != 0 else "Female"
    governorate = governorates.get(governorate_code, "Unknown")
    birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"

    return {"Birth Date": birth_date, "Governorate": governorate, "Gender": gender}


# Function to detect the ID card and pass it to the existing code
def detect_and_process_id_card(image):
    # Load the ID card detection model
    # id_card_model = YOLO("detect_id_card.pt")

    # Perform inference to detect the ID card
    id_card_results = id_card_model(image)

    # Crop the ID card from the image
    for result in id_card_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]

    # Pass the cropped image to the existing processing function
    return process_image(cropped_image)


def detect_and_process_id_card_frame(frame):
    return process_image(frame)
