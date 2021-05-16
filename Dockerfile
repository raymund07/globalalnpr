

FROM python:3.8


RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY classes /classes
COPY inferencegraphs /inferencegraphs
COPY received /received
COPY application /application

WORKDIR ./application
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
    
CMD ["python","server.py"]

