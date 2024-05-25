from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import warnings
import pickle
import os
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt()
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

app = Flask(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///phishing_predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define a model for storing URLs and predictions
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(200), nullable=False)
    legitimate_suggestion = db.Column(db.String(200), nullable=True)

# Load the pre-trained model
model_file_path = "model.pkl"
with open(model_file_path, "rb") as file:
    gbc = pickle.load(file)

# File to store phishing URLs
phishing_file = "phishing_websites.txt"

# Read mapping from Excel file
mapping_file_path = "Mapping.xlsx"
mapping_df = pd.read_excel(mapping_file_path, engine='openpyxl', header=None)
phishing_to_legitimate = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))

@app.route('/')
def first():
    return render_template('first.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST", "GET"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)

    default_csv_path = "upload.csv"
    df_default = pd.read_csv(default_csv_path, encoding='unicode_escape')
    df_default.set_index('Id', inplace=True)
    return render_template("preview.html", df_view=df_default)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/posts", methods=["GET", "POST"])
def posts():
    legitimate_suggestion = None

    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

        prediction = "It is {100:.2f} % safe to go ".format(y_pro_phishing * 100) if y_pred == 1 else "The website is detected as phishing and not safe to go."

        # Store the prediction in the database
        new_prediction = Prediction(url=url, prediction=prediction, legitimate_suggestion=legitimate_suggestion)
        db.session.add(new_prediction)
        db.session.commit()

        # Store the phishing website in a file
        with open(phishing_file, "a") as f:
            f.write(url + "\n")

        # Check if there's a legitimate suggestion for the phishing site
        legitimate_suggestion = phishing_to_legitimate.get(url)
        if legitimate_suggestion:
            prediction += f"\nYou might want to visit the legitimate site: {legitimate_suggestion}"

        return render_template('result.html', xx=round(y_pro_non_phishing, 2), url=url, prediction=prediction,
                               legitimate_suggestion=legitimate_suggestion)

    return render_template("result.html", xx=-1)

@app.route('/all_predictions')
def all_predictions():
    predictions = Prediction.query.all()
    return render_template('all_predictions.html', predictions=predictions)

if __name__ == "__main__":
    with app.app_context():
        # Create the database tables before running the app
        db.create_all()

    app.run(debug=True)
