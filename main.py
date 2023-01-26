from flask import *
import os
import pickle 
app = Flask(__name__)
import tensorflow as tf 
from  tensorflow.keras.layers import TextVectorization
import  numpy as np

#ml  vector  loading
from_disk=pickle.load( open('models/vector_layer.pkl','rb') )
new_vectorizer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                          output_mode='int',
                                          output_sequence_length=from_disk['config']['output_sequence_length'])
# new_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_vectorizer.set_weights(from_disk['weights'])

# ml  model loading
model=tf.keras.models.load_model('models/toxicity.h5')


@app.route('/')
def index():
    return jsonify({"USER": "This is  backend server for comments toxicity prediction using deep learning "})


# route  for prediction
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=="POST":
        comments=request.form['comments']
        input_str=new_vectorizer(comments)
        # print(input_str)
        res=model.predict(np.expand_dims(input_str,0))
        # print(res)
        return jsonify({"toxic":str(res[0][0]),
                        "severe_toxic":str(res[0][1]),
                        "obscene":str(res[0][2]),
                        "threat":str(res[0][3]),
                        "insult":str(res[0][4]),
                        "identity_hate":str(res[0][5])})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))