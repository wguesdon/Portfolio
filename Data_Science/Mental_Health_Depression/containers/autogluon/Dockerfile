FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY Data_Science/Mental_Health_Depression/containers/autogluon/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# AutoGluon trainer — standalone, does not use base_trainer
COPY Data_Science/Mental_Health_Depression/src/train_autogluon.py .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "train_autogluon.py"]
