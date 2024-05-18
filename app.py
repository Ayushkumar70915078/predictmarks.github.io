import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib as jb

app = Flask(__name__)

# Load the model
model = jb.load("students_marks_predictior_model.pkl")

# Initialize a global DataFrame to store the data
df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global df
    
    # Retrieve input features from the form
    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features).reshape(1, -1)
    
    # Validate input hours
    if input_features[0] < 0 or input_features[0] > 24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
    
    # Make prediction
    output = model.predict(features_value).flatten()[0].round(2)

    # Append input and predicted values to the DataFrame
    df = pd.concat([df, pd.DataFrame({'Study Hours': [input_features[0]], 'Predicted Output': [output]})], ignore_index=True)
    print(df)
    
    # Save the DataFrame to a CSV file
    df.to_csv('smp_data_from_app.csv', index=False)

    return render_template('index.html', prediction_text='You will get [{}%] marks, when you do study [{}] hours per day'.format(output, input_features[0]))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
