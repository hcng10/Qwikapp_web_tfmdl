FROM tensorflow/tensorflow:2.15.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --ignore-installed blinker
RUN pip install --no-cache-dir Flask

EXPOSE 5000

WORKDIR /usr/src/app
CMD python app.py
