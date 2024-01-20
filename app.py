from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates')


# Load model and scaler outside of the request handling function
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['GET','POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]

    try:
        pre_final_features = np.array(int_features).reshape(1, -1)
        final_features = scaler.transform(pre_final_features)

        prediction = model.predict(final_features)

        if prediction[0] == 1:
            output = "This person is diabetic."
        elif prediction[0] == 0:
            output = "The person is not diabetic."
        else:
            output = "Not sure."

    except Exception as e:
        output = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=f'=> {output}')

if __name__ == "__main__":
    app.run(debug=True)
