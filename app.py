from tensorflow import keras
from flask import Flask, request, jsonify
import numpy as np
# Create Flask app
app = Flask(__name__)

# Load model from H5 file


# Define endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    request_data = request.get_json()

    # Check if the 'data' field is present in the request data
    if 'data' not in request_data:
        return jsonify({"error": "Missing 'data' field"}), 422
    if 'id' not in request_data:
        return jsonify({"error": "Missing 'id' field"}), 422
    # Get the 'data' array from the request data
    id = request_data['id']
    data = request_data['data']

    # Check the length of the 'data' array
    expected_length = 1000  # Modify this to the expected length
    if len(data) != expected_length:
        return jsonify({"error": f"Expected 'data' array to have length {expected_length}, got {len(data)}"}), 422
    model = keras.models.load_model('data/'+id+'.h5')
    # Preprocess input data if required
    data = np.array(data)
    data = (data - data.min()) / (data.max() - data.min())
    data = np.reshape(data, (1, data.shape[0], 1))
    # Make predictions using loaded model
    predictions = model.predict(data)

    # Postprocess predictions if required
    # ...

    # Return predictions as JSON response
    return jsonify(predictions.tolist()[0])

# Start Flask app
if __name__ == '__main__':
    app.run()