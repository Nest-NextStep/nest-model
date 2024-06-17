import requests

# Define the URL of your Flask app
url = "http://127.0.0.1:5000/predict"

# Define the JSON data to send
data = {
    "R1": 5, "R2": 5, "R4": 5, "R6": 4, "R7": 5, "R8": 3,
    "I1": 1, "I2": 2, "I4": 3, "I5": 4, "I7": 5, "I8": 3,
    "A2": 1, "A3": 2, "A4": 3, "A5": 4, "A6": 5, "A8": 3,
    "S1": 1, "S3": 2, "S5": 3, "S6": 4, "S7": 5, "S8": 3,
    "E1": 1, "E3": 2, "E4": 3, "E5": 4, "E7": 5, "E8": 3,
    "C2": 1, "C3": 2, "C5": 3, "C6": 4, "C7": 5, "C8": 3,
    "TIPI1": 1, "TIPI2": 2, "TIPI3": 3, "TIPI4": 4, "TIPI5": 5,
    "TIPI6": 6, "TIPI7": 7, "TIPI8": 4, "TIPI9": 4, "TIPI10": 6,
    "VCL1": 1, "VCL2": 1, "VCL3": 0, "VCL4": 1, "VCL5": 0,
    "VCL6": 1, "VCL10": 0, "VCL11": 1, "VCL12": 1, "VCL13": 1,
    "VCL14": 1, "VCL15": 1, "education": 1, "gender": 1,
    "engnat": 1, "religion": 1, "voted": 1
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
