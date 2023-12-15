FROM python:3.11.3

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install gunicorn
COPY . .

CMD exec gunicorn -w 4 -b 0.0.0.0:5000 api:app