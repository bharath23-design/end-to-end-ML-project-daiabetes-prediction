from flask import Flask, request, render_template
import mysql.connector
import numpy as np
import joblib
from urllib.parse import quote

app = Flask(__name__)

# Connect to MySQL using MySQL Connector
def connect_to_database():
    password = 'balu1234'
    encoded_password = quote(password)
    
    db = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password=encoded_password,
        database='diabetes',
        port=3306
    )
    
    return db

def create_table_if_not_exists(db):
    cursor = db.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FormData (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pregnancies INT,
            glucose INT,
            blood_pressure INT,
            skin_thickness INT,
            insulin INT,
            bmi FLOAT,
            diabetes_pedigree_function FLOAT,
            age INT,
            prediction VARCHAR(1000)
        )
    ''')
    db.commit()

# Load model and scaler outside of the request handling function
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Initialize db outside of the functions to make it global
db = connect_to_database()
create_table_if_not_exists(db)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    global db  # Declare db as global
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

        # Store form data and prediction in the database
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO FormData
            (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, prediction)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (int_features[0], int_features[1], int_features[2], int_features[3], int_features[4],
              int_features[5], int_features[6], int_features[7], output))

        db.commit()

    except Exception as e:
        output = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=f'=> {output}')

@app.route('/view_data', methods=['GET'])
def view_data():
    global db  # Declare db as global
    try:
        cursor = db.cursor()
        cursor.execute('SELECT * FROM FormData')
        data = cursor.fetchall()
        column_names = [i[0] for i in cursor.description]
    except Exception as e:
        return f"Error: {str(e)}"

    return render_template('view_data.html', data=data, column_names=column_names)

if __name__ == "__main__":
    app.run(debug=True)
