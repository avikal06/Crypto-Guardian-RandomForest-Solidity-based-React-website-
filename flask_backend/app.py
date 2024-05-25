from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import PowerTransformer

app = Flask(__name__)

# Load the pre-trained model and normalizer
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/power_transformer.pkl', 'rb') as norm_file:
    norm = pickle.load(norm_file)

@app.route('/check_blacklist', methods=['POST'])
def check_blacklist():
    data = request.get_json()
    account = data['account']

    # Dummy features for demonstration
    features = pd.DataFrame([{
        'feature1': 0.1,
        'feature2': 0.2,
        # Add all required features here
    }])

    # Normalize the features
    normalized_features = norm.transform(features)

    # Predict using the loaded model
    is_blacklisted = model.predict(normalized_features)[0] == 1

    return jsonify({'blacklisted': bool(is_blacklisted)})

if __name__ == '__main__':
    app.run(debug=True)
