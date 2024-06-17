import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("MODEL_2.h5", compile=False)

labels = {
    0: 'administrasi bisnis', 1: 'akuntansi', 2: 'antropologi', 3: 'arsitektur', 4: 'bimbingan konseling', 5: 'biologi', 6: 'desain grafis', 7: 'digital bisnis', 8: 'ekonomi', 9: 'farmasi', 10: 'filsafat', 11: 'fisika', 12: 'geografi', 13: 'hubungan internasional', 14: 'hukum', 15: 'ilmu kesejahteraan sosial', 16: 'ilmu komunikasi', 17: 'ilmu politik', 18: 'jurnalistik', 19: 'kedokteran', 20: 'keperawatan', 21: 'kesehatan masyarakat', 22: 'kimia', 23: 'kriminologi', 24: 'linguistik', 25: 'manajemen', 26: 'manajemen bisnis', 27: 'hubungan msyarakat', 28: 'marketing', 29: 'matematika', 30: 'musik', 31: 'pariwisata', 32: 'pendidikan olahraga', 33: 'Pendidikan Anak Usia Dini', 34: 'psikologi', 35: 'sastra inggris', 36: 'sejarah', 37: 'seni rupa', 38: 'sosiologi', 39: 'teknik biomedik', 40: 'teknik elektro', 41: 'teknik elektronika', 42: 'teknik industri', 43: 'teknik informatika', 44: 'teknik kimia', 45: 'teknik komputer', 46: 'teknik lingkungan', 47: 'teknik mesin', 48: 'teknik sipil', 49: 'teknologi informasi'
}

R_features = ['R1', 'R2', 'R4', 'R6', 'R7', 'R8']
I_features = ['I1', 'I2', 'I4', 'I5', 'I7', 'I8']
A_features = ['A2', 'A3', 'A4', 'A5', 'A6', 'A8']
S_features = ['S1', 'S3', 'S5', 'S6', 'S7', 'S8']
E_features = ['E1', 'E3', 'E4', 'E5', 'E7', 'E8']
C_features = ['C2', 'C3', 'C5', 'C6', 'C7', 'C8']


@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 201,
            "message": "API is running"
        },
        "data": None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.json
            # Ensure all required features are in the input data
            required_features = R_features + I_features + A_features + S_features + E_features + C_features + [
                'TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8', 'TIPI9', 'TIPI10',
                'VCL1', 'VCL2', 'VCL3', 'VCL4', 'VCL5', 'VCL6', 'VCL10', 'VCL11', 'VCL12', 'VCL13', 'VCL14', 'VCL15',
                'education', 'gender', 'engnat', 'religion', 'voted'
            ]

            for feature in required_features:
                if feature not in data:
                    return jsonify({
                        "status": {
                            "code": 400,
                            "message": f"Feature '{feature}' is missing from input data"
                        },
                        "data": None
                    }), 400

            # Convert data to DataFrame for easier manipulation
            df = pd.DataFrame([data])

            # Sum the feature groups
            df['R'] = df[R_features].sum(axis=1)
            df['I'] = df[I_features].sum(axis=1)
            df['A'] = df[A_features].sum(axis=1)
            df['S'] = df[S_features].sum(axis=1)
            df['E'] = df[E_features].sum(axis=1)
            df['C'] = df[C_features].sum(axis=1)

            # Select the final 37 features
            final_features = ['R', 'I', 'A', 'S', 'E', 'C'] + [
                'TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8', 'TIPI9', 'TIPI10',
                'VCL1', 'VCL2', 'VCL3', 'VCL4', 'VCL5', 'VCL6', 'VCL10', 'VCL11', 'VCL12', 'VCL13', 'VCL14', 'VCL15',
                'education', 'gender', 'engnat', 'religion', 'voted'
            ]
            input_data = df[final_features].values

            # Predict using the model
            prediction = model.predict(input_data)
            y_pred = np.argmax(prediction, axis=1)

            y_pred_labels = [labels[i] for i in y_pred]

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Prediction successful"
                },
                "data": {
                    "prediction": y_pred_labels
                }
            })
        except Exception as e:
            return jsonify({
                "status": {
                    "code": 500,
                    "message": str(e)
                },
                "data": None
            }), 500

    return jsonify({
        "status": {
            "code": 405,
            "message": "Method not allowed"
        },
        "data": None
    }), 405


if __name__ == "__main__":
    app.run(debug=True)
