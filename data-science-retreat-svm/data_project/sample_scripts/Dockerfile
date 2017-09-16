FROM python:2-slim

MAINTAINER Diego Amicabile <diego.amicabile@gmail.com>

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

CMD ["bash",  "update_news.sh"]



