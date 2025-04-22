FROM python:3.12-bookworm

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
