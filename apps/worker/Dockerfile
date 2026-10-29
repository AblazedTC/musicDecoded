FROM python:3.9-slim

# System dependencies (ffmpeg for audio, libsndfile for audio file reading,
# gcc/g++ for any dependency that has no prebuilt wheel)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The consumer validates every analysis against the contract before saving it.
# `contracts` is a named build context (docker-compose.yml additional_contexts);
# services/consumer.py finds it at /contracts.
COPY --from=contracts analysis.schema.json validate.py /contracts/

ENV PYTHONUNBUFFERED=1
# BLPOP jobs:analysis -> analyze -> analysis/{trackId}.json (T-113)
CMD ["python", "services/consumer.py"]
