# pull python base image
FROM python:3.10

# copy application files
ADD . .

# specify working directory
WORKDIR /app

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# start fastapi application
CMD ["python", "adultcensus_model/train_pipeline.py"]