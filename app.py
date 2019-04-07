#MINOR
from flask import Flask, request, render_template
import pandas as pd
from shapely.geometry import Point, shape
from flask import Flask
from flask import render_template
import json
import pickle
import numpy as np
import pandas as pd


data_path = './input/'




app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")



@app.route('/getpred',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
        
        pkl_file = open('cat', 'rb')
        index_dict = pickle.load(pkl_file)
        new_vector = np.zeros(len(index_dict))
        
        
        try:
            new_vector[index_dict['Cyl_'+str(result['Cyl'])]] = 1
        except:
            pass
			
        try:
            new_vector[index_dict['Drive_'+str(result['Drive'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Fuel_'+str(result['Fuel'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Veh_Class_'+str(result['Veh_Class'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Air_Pollution_Score_'+str(result['Air_Pollution_Score'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['SmartWay_'+str(result['SmartWay'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['year_'+str(result['year'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Transmission_type_'+str(result['Transmission_type'])]] = 1
        except:
            pass
        
			
        
        pkl_file = open('logmodel.pkl', 'rb')
        logmodel = pickle.load(pkl_file)
        prediction = logmodel.predict(new_vector.reshape(1, -1))
	   
        return render_template('result.html',prediction=prediction)
		
@app.route('/getpred1',methods=['POST','GET'])
def get_delay_1():
    if request.method=='POST':
        result=request.form
        
        pkl_file = open('cat_mpg', 'rb')
        index_dict = pickle.load(pkl_file)
        new_vector = np.zeros(len(index_dict))
        
        
        try:
            new_vector[index_dict['Cyl_'+str(result['Cyl'])]] = 1
        except:
            pass
			
        try:
            new_vector[index_dict['Drive_'+str(result['Drive'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Fuel_'+str(result['Fuel'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Veh_Class_'+str(result['Veh_Class'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Air_Pollution_Score_'+str(result['Air_Pollution_Score'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['SmartWay_'+str(result['SmartWay'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['year_'+str(result['year'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Transmission_type_'+str(result['Transmission_type'])]] = 1
        except:
            pass
        
			
        
        pkl_file = open('logmodel_MPG.pkl', 'rb')
        logmodel = pickle.load(pkl_file)
        prediction = logmodel.predict(new_vector.reshape(1, -1))
	   
        return render_template('result_1.html',prediction=prediction)


@app.route("/data")
def get_data():
    
	
    df_clean = pd.read_csv(data_path + 'OUT_1.csv')
	
    

    return df_clean.to_json(orient='records')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)