# pull python base image
FROM python:3.10

# copy application files
ADD . .

# specify working directory
WORKDIR /app

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements/requirements.txt

# start training model
CMD ["python", "adultcensus_model/train_pipeline.py"]