FROM python:3.9-slim

# System dependencies (ffmpeg for audio, libsndfile for audio file reading,
# gcc/g++ needed to compile native extensions like madmom)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cython and numpy must be installed before madmom (it builds C extensions at install time)
RUN pip install --no-cache-dir Cython numpy

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Placeholder: keeps the container running until a proper HTTP/queue entrypoint is added
CMD ["python", "-c", "import time; print('Worker ready'); [time.sleep(60) for _ in iter(int, 1)]"]
