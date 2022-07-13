FROM ubuntu:latest

RUN apt-get update -y

RUN apt-get install python3 -y

RUN apt-get install python3-pip -y

WORKDIR /home

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--reload"]