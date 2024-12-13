# Imagen base
FROM python:3.9-slim

WORKDIR /app

COPY frontend.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "frontend.py"]
