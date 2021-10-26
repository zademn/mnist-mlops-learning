FROM python:3.9-slim

RUN mkdir /frontend

COPY requirements.txt /frontend

WORKDIR /frontend

RUN pip install -r requirements.txt

COPY . /frontend

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_main.py"]


