'''
References:
Table: https://physionet.org/content/gait-maturation-db/1.0.0/data/table.csv
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
https://joblib.readthedocs.io/en/latest/why.html
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
https://towardsdatascience.com/using-joblib-to-speed-up-your-python-pipelines-dd97440c653d
https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4
'''
from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np
import uuid
import plotly.express as px
import plotly.graph_objects as go
import os

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template("main.html")


@app.route('/dashboard/', methods=['GET', 'POST'])
def dashboard():
    data_frame = data_exploration()
    req_type = request.method
    if req_type == 'GET':
        return render_template("dashboard.html", href='static/images/base_pic.svg', tables=[data_frame.to_html(classes='data')], titles=data_frame.columns.values)
    else:
        ml_model()
    return render_template("dashboard.html")


def data_exploration():
    # data = read_csv('table.csv')
    data = pd.read_pickle('AgesAndHeights.pkl')
    data_frame = pd.DataFrame(data)
    return data_frame.sample(n=10, random_state=1)


def ml_model():
    text = request.form['text']
    random_string = uuid.uuid4().hex
    rel_path = "static/images"
    r_string = f"{random_string}" + ".svg"
    path = os.path.join(rel_path, r_string)
    model = load('model.joblib')
    np_arr = float_str_np_array(text)
    make_picture('AgesAndHeights.pkl', model, np_arr, path)
    return render_template('dashboard.html', href=path[4:])


def make_picture(file, model, np_arr, path):
    data = pd.read_pickle(file)
    ages = data['Age']
    data = data[ages > 0]
    ages = data['Age']
    heights = data['Height']
    x_new = np.array(list(range(19))).reshape(19, 1)
    preds = model.predict(x_new)
    fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={
                     'x': 'Age (years)', 'y': 'Height (inches)'})
    fig.add_trace(go.Scatter(x=x_new.reshape(
        19), y=preds, mode='lines', name='Model'))
    new_preds = model.predict(np_arr)
    fig.add_trace(go.Scatter(x=np_arr.reshape(len(np_arr)), y=new_preds, name='New Outputs',
                  mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))
    fig.write_image(path, width=800, engine='kaleido')
    fig.show()


def float_str_np_array(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)
