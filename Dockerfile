FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && (pip uninstall -y py-key-value-shared || true) \
    && pip install --force-reinstall py-key-value-aio==0.4.4

COPY . /app

EXPOSE 8080

CMD ["python", "server.py"]
