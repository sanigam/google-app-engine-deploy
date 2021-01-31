from flask import Flask, request
import pickle
#from google.cloud import storage
import numpy as np
#from sklearn.linear_model import LogisticRegression

# storage_client = storage.Client()
# bucket = storage_client.get_bucket('model-bucket-iris-sn')
# blob = bucket.blob('flower-v1.pkl')
# blob.download_to_filename('/tmp/flower-v1.pkl')
model_pk = pickle.load(open('flower-v1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/api_predict', methods=['GET', 'POST'])
def api_predict():
    if request.method == 'GET':
        return "Please Send POST Request"
    elif request.method == 'POST':
        data = request.get_json()
        
        print("Data sent:" , str(data))
        
        
        sepal_length = data["sepal_length"]
        sepal_width = data["sepal_width"]
        petal_length = data["petal_length"]
        petal_width = data["petal_width"]
    
        data = np.array([[sepal_length, sepal_width, 
                          petal_length, petal_width]])
           
        prediction = model_pk.predict(data)
        return str(prediction)

