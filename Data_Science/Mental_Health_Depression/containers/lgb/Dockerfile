FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY Data_Science/Mental_Health_Depression/containers/lgb/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared base trainer from aws/sagemaker/src/
COPY aws/sagemaker/src/base_trainer.py .
# Project-specific trainer
COPY Data_Science/Mental_Health_Depression/src/train_lgb.py .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "train_lgb.py"]
