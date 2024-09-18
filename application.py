from flask import Flask, render_template, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Load model and data
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

car_data = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car_data['company'].unique())
    car_models = sorted(car_data['name'].unique())
    years = sorted(car_data['year'].unique(), reverse=True)
    fuel_types = car_data['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    # Prepare the input data for prediction
    input_data = pd.DataFrame([[car_model, company, year, driven, fuel_type]], 
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    try:
        # Predict using the loaded model
        prediction = model.predict(input_data)
    except Exception as e:
        return f"Error during prediction: {e}"  # Return error message for debugging

    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)