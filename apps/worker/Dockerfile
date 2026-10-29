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
# CPU-only torch: the default PyPI wheel pulls ~3 GB of CUDA libraries we don't use.
# ponytail: CPU inference; switch the index to a CUDA one for a GPU worker (Pro models, T-115).
# --extra-index-url (not --index-url): everything else still comes from PyPI; torch resolves to
# the higher-ranked "+cpu" build from the PyTorch index.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# The consumer validates every analysis against the contract before saving it.
# `contracts` is a named build context (docker-compose.yml additional_contexts);
# services/consumer.py finds it at /contracts.
COPY --from=contracts analysis.schema.json validate.py /contracts/

ENV PYTHONUNBUFFERED=1
# BLPOP jobs:analysis -> analyze -> analysis/{trackId}.json (T-113)
CMD ["python", "services/consumer.py"]
