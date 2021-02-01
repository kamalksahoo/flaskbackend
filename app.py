from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin


import json
import tensorflow as tf
import numpy as np
import cv2


app = Flask(__name__)        
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# POST - just get the image and metadata
@app.route('/RequestImageWithMetadata', methods=['POST'])
def post():
    #request_data = request.form['some_text']
    #print(request_data)
    imagefile = request.files.get('imagefile', '')
    #print(request.files)
    
    imagefile.save('./test_image.png')

    
    model = tf.keras.models.load_model('./classification_model.h5')
    #model = tf.keras.models.load_model('E://downloads//classification.h5')

    img_array = cv2.imread('./test_image.png',0)

    img_array = cv2.resize(img_array, (50,50)) 

    img_array = img_array / 255

    img_array = img_array.reshape(1,50,50,1)

    prediction = model.predict(img_array)
    
    return jsonify({"result":str(prediction[0][0])}), 200

app.run(port=5000)
