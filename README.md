# **Egyptian ID Card Recognition – AI-Powered OCR**

**A Python-based AI application for detecting, processing, and extracting fields from Egyptian ID cards using YOLO, Tesseract,and EasyOCR.**

## **Key Features**

**AI-Powered ID Detection** – Automatically detects and crops Egyptian ID cards from images.  
**Optical Character Recognition** – Extracts typed Arabic text from ID cards using EasyOCR & Tesseract.  
**Field Extraction & Data Processing** – Captures essential details, including:

- First Name
- Last Name
- Address
- National ID Number
- Birth Date
- Governorate
- Gender
- Birth Place

## **Installation Guide**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aabolfadl/egyptian-id-card-OCR.git
   ```
3. **Navigate to the project directory**:

   ```bash
   cd egyptian-id-card-OCR
   ```

4. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   ```

   Creating the virtal environment (first line) is to be run the first time only
   Activating the virtual environment (second line) is to be run whenever you open the development environment (e.g. VS Code)

5. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Paste your id image in test_ids/**:
   Supported file types: jpeg, jpg, png

7. **Update main.py **:
   Edit `line 3` with your image file's name

8. **Run the application in terminal**:
   ```powershell
   (venv) path\to\your\development\directory> python main.py
   ```

## **Model Training**

- **YOLO Object Detection** – Trained for Egyptian ID card detection.
- **EasyOCR** – Used for high-accuracy text recognition in Arabic.
- **Tesseract** – Used for high-accuracy text recognition in Arabic.

# Packages tried
- **EasyOCR** – Used for high-accuracy text recognition in Arabic.
- **Tesseract** – Used for high-accuracy text recognition in Arabic for First Name (EasyOCR performs badly on short strings).
- **PaddleOCR** - Bad performance