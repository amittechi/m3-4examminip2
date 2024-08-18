FROM python:3.9-slim

WORKDIR /app

COPY requirements/requirements.txt requirements/requirements.txt
RUN pip install --no-cache-dir -r requirements/requirements.txt

COPY . .

CMD ["python", "adultcensus_model/train_pipeline.py"]
