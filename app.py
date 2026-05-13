import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import os
import logging

# --- IMPORT AZURE MONITOR & OPENTELEMETRY ---
from azure.monitor.opentelemetry import configure_azure_monitor

app = Flask(__name__)

# --- INITIALIZE AZURE TELEMETRY ---
# This pulls the connection string dynamically from your Azure App Service environment variables
AZURE_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

if AZURE_CONNECTION_STRING:
    configure_azure_monitor(connection_string=AZURE_CONNECTION_STRING)
    # Configure a logger that sends data directly to your Log Analytics Workspace
    logger = logging.getLogger("manas")
    logger.setLevel(logging.INFO)
    logger.info("Flask App successfully linked to Azure Monitor telemetry pipeline.")
else:
    # Fallback to standard console logging if running locally without Azure
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("manas")
    logger.warning("APPLICATIONINSIGHTS_CONNECTION_STRING not found. Running with local logs only.")


# Load your machine learning model
model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI and streaming data to Azure
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # 1. Log to Azure that a prediction workflow has started
    logger.info("Starting MLOps prediction execution step", extra={
        "custom_dimensions": {
            "WorkflowName": "InsuranceChargesWorkflow",
            "ExecutorType": "ModelPredictionNode",
            "FeaturesCount": str(len(int_features))
        }
    })

    # Run the model prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    # 2. Log the successful completion and the output value
    logger.info(f"Prediction complete. Output: {output}", extra={
        "custom_dimensions": {
            "WorkflowName": "InsuranceChargesWorkflow",
            "PredictionStatus": "Success",
            "PredictedCharges": str(output)
        }
    })

    return render_template('index.html', prediction_text='Insurance Charges {}'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
