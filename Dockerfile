FROM python:3.9.18-slim

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY . /app

WORKDIR /app

RUN apt-get update && apt-get install -y tesseract-ocr

EXPOSE 8080

CMD ["python", "main.py"]
