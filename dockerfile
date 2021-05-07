FROM python:3.7.1

WORKDIR .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["flask", "run"]