FROM tensorflow/tensorflow:2.15.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#USER nobody

COPY . /usr/src/app
WORKDIR /usr/src/app
CMD python run.py
