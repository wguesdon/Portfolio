FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /opt/ml/code

COPY Data_Science/Obesity_Risk_Prediction/containers/xgb/requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Shared base trainer from aws/sagemaker/src/
COPY aws/sagemaker/src/base_trainer.py .
# Project-specific trainer
COPY Data_Science/Obesity_Risk_Prediction/src/train_xgb.py .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "train_xgb.py"]
