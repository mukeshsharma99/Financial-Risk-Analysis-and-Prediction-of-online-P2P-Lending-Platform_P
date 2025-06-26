from flask import Flask, render_template, request
import pandas as pd
import joblib

application = Flask(__name__)

# Load the saved pipelines and LabelEncoder
classification_pipeline = joblib.load('classification_pipeline.pkl')
regression_pipeline_ela = joblib.load('regression_pipeline_ela.pkl')
regression_pipeline_emi = joblib.load('regression_pipeline_emi.pkl')
label_encoder = joblib.load('label_encoder.pkl')  

# Define a mapping for loan status
loan_status_mapping = {0: 'Not Approved', 1: 'Approved'}

@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        emi = float(request.form.get('emi'))
        ela = float(request.form.get('ela'))
        roi = float(request.form.get('roi'))

        input_data = pd.DataFrame([[emi, ela, roi]], columns=['EMI', 'ELA', 'ROI'])

        # Classification prediction
        loan_status_pred = classification_pipeline.predict(input_data)[0]
        loan_status_label = loan_status_mapping[loan_status_pred]

        # Regression predictions
        ela_pred = regression_pipeline_ela.predict(input_data)[0]
        emi_pred = regression_pipeline_emi.predict(input_data)[0]

        return render_template('index.html', 
                               loan_status=loan_status_label, 
                               ela_pred=f"{ela_pred:.2f}", 
                               emi_pred=f"{emi_pred:.2f}")

    return render_template('index.html', loan_status=None, ela_pred=None, emi_pred=None)

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8080)
