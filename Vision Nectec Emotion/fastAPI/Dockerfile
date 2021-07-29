FROM python:3.8
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt .
RUN apt update
RUN apt-get install -y libhdf5-dev
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
COPY ./app /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]