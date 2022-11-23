import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(pclass, sex, age_bin, fare_bin):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age_bin)
    input_list.append(fare_bin)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    res_0 = str(res[0])
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    prediction_url = "https://raw.githubusercontent.com/torileatherman/serverless_ml_titanic/main/src/assets/"+res_0+".png"
    img = Image.open(requests.get(prediction_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analytics",
    description="Experiment with class, sex, age, and fare type to predict if the passenger survived",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1, label="Class (1 is highest, 3 is lowest"),
        gr.inputs.Number(default=1, label="Gender (0 is male, 1 is female)"),
        gr.inputs.Number(default=20, label="Age (years)"),
        gr.inputs.Number(default=1, label="Fare Type (1 is lowest, 4 is highest)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()