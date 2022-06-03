import pickle
import numpy as np
from sklearn import preprocessing
from flask import Flask, request, render_template

# Initialize the flask class and specify the templates directory
app = Flask(__name__, template_folder="template")

model = pickle.load(open('saved_model.pkl', 'rb'))


# Create our "home" route using the "index.html" page
@app.route('/')
def home():
    return render_template('index.html')


# Route 'predict'
@app.route('/', methods=['POST'])
def predict():
    try:
        features = [[float(x) for x in request.form.values()]]

        norm = preprocessing.Normalizer().fit(features)
        features = norm.transform(features)
        features = np.asarray(features)

        # Get the output from the model
        prediction = model.predict(features)

        # Render the output in new HTML page
        return render_template('index.html', prediction_text=prediction)
    except:
        return 'Error'


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
