# -*- coding: utf-8 -*-
"""
Created on Mon May  2 22:04:33 2022

@author: manju
"""
import flask
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app=flask(_name_)
filename = 'model_gym.pkl'
model=pickle.load(open(filename,'rb'))
@app.route('/')
def man():
    return render_template('home.html')
app.route('/predict', methods=['POST'])
def home():
    data1=request.form['a']
    data2=request.form['b']
    data3=request.form['c']
    arr=np.array([[data1, data2, data3]])
    pred=model.predict(arr)
    return render_template('after.html', data=pred)

if__name__ == "__main__" :
    app.run(debug=True)
    







