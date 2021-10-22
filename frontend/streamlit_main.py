import numpy as np
from numpy.lib.function_base import select
import pandas as pd

import time
import streamlit as st
from streamlit.state.session_state import SessionState
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

import cv2
import requests
import urllib
import json


# Configs
MODEL_INPUT_SIZE = 28
CANVAS_SIZE = MODEL_INPUT_SIZE * 8
BACKEND_URL = "http://localhost:8000"
MODELS_URL = urllib.parse.urljoin(BACKEND_URL, "models")
TRAIN_URL = urllib.parse.urljoin(BACKEND_URL, "train")
PREDICT_URL = urllib.parse.urljoin(BACKEND_URL, "predict")
DELETE_URL = urllib.parse.urljoin(BACKEND_URL, "delete")


st.title("Mnist training and prediction")
st.sidebar.subheader("Page navigtion")
page = st.sidebar.selectbox(label="", options=[
    "Train", "Predict", "Delete"])
st.sidebar.write("https://github.com/zademn")

if page == "Train":
    # Conv is not provided yet
    st.session_state.model_type = st.selectbox(
        "Model type", options=["Linear", "Conv"])

    model_name = st.text_input(label="Model name", value="My Model")

    if st.session_state.model_type == "Linear":
        num_layers = st.select_slider(
            label="Number of hidden layers", options=[1, 2, 3])
        cols = st.columns(num_layers)
        hidden_dims = [64] * num_layers
        for i in range(num_layers):
            hidden_dims[i] = cols[i].number_input(
                label=f"Number of neurons in layer {i}", min_value=2, max_value=128, value=hidden_dims[i])

        hyperparams = {
            "input_dim": 28 * 28,
            "hidden_dims": hidden_dims,
            "output_dim": 10,
        }

        epochs = st.number_input("Epochs", min_value=1, value=5, max_value=128)

    if st.button("Train"):
        st.write(f"{hyperparams=}")
        to_post = {"model_name": model_name,
                   "hyperparams": hyperparams, "epochs": epochs}
        response = requests.post(url=TRAIN_URL, data=json.dumps(to_post))
        if response.ok:
            res = response.json()["result"]
        else:
            res = "Training task failed"
        st.write(res)

    # if st.session_state.model_type == "Conv":
    #     pass

elif page == "Predict":

    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")
    # Setup canvas
    canvas_res = st_canvas(
        fill_color="black",  # Black
        stroke_width=20,
        stroke_color="white",  # White
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key='canvas',
        display_toolbar=True
    )

    # Get image
    if canvas_res.image_data is not None:
        # Scale down image to the model input size
        img = cv2.resize(canvas_res.image_data.astype("uint8"),
                         (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        # Rescaled image upwards to show
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rescaled = cv2.resize(
            img, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
        st.write("Model input")
        st.image(img_rescaled)

    # Predict on the press of a button
    if st.button("Predict"):
        try:
            response_predict = requests.post(url=PREDICT_URL,
                                             data=json.dumps({"input_image": img.tolist(), "model_name": model_name}))
            if response_predict.ok:
                pred = response_predict.json()
                st.write(pred)
            else:
                st.write("Some error occured")
        except ConnectionError as e:
            st.write("Couldn't reach backend")

elif page == "Delete":
    try:
        response = requests.get(MODELS_URL)
        if response.ok:
            model_list = response.json()
            model_name = st.selectbox(
                label="Select your model", options=model_list)
        else:
            st.write("No models found")
    except ConnectionError as e:
        st.write("Couldn't reach backend")

    to_post = {"model_name": model_name}
    # Delete on the press of a button
    if st.button("Delete"):
        try:
            response = requests.post(url=DELETE_URL,
                                     data=json.dumps(to_post))
            if response.ok:
                res = response.json()
                st.write(res)
            else:
                st.write("Some error occured")
        except ConnectionError as e:
            st.write("Couldn't reach backend")
else:
    st.write("Page does not exist")
