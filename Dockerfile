FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir numpy==1.26.4

RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt

# Download model during build
RUN mkdir -p model && \
    wget -q --no-check-certificate \
    "https://drive.google.com/uc?export=download&id=1Wg3NcdCTOh-xYf6hkP0bSQkqwC2PIvH5" \
    -O model/efficientnet_indian_food.pth && \
    echo "Model downloaded!"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]