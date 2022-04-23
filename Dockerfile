FROM python:3.8-slim
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
COPY ./src /app
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["App.py"]