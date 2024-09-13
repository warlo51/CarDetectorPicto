# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.6

FROM python:${PYTHON_VERSION}-slim

LABEL fly_launch_runtime="flask"

WORKDIR /code

# Installer les dépendances systèmes nécessaires pour OpenCV et nettoyer après
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]
