FROM python:3.12
WORKDIR /app
# System deps for OpenCV, Tesseract, pdf2image, Jupyter, zmq
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libtesseract-dev \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    libzmq3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install -r requirements.txt -v --progress-bar=raw
COPY app/ ./app/
COPY artifacts/ ./artifacts/
EXPOSE 8080
CMD ["uvicorn", "app.api:main", "--host", "0.0.0.0", "--port", "8080"]