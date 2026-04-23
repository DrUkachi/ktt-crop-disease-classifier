FROM python:3.11-slim

WORKDIR /app

# System deps: libgomp for onnxruntime, libjpeg/libpng for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY service/app.py /app/service/app.py
COPY model.onnx    /app/model.onnx

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
