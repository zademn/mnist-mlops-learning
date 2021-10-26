FROM python:3.9-slim

RUN pip install mlflow

EXPOSE 5000

CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///db/backend.db", "--host", "0.0.0.0"]


