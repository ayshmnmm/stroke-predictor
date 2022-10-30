import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# Read the data
dataset = pd.read_csv('stroke.csv')

# convert floating point values to integer
dataset['age'] = dataset['age'].astype(int)

# label encoding
leg = LabelEncoder()
lem = LabelEncoder()
lew = LabelEncoder()
ler = LabelEncoder()
les = LabelEncoder()
leg.fit(dataset.gender)
dataset['gender'] = leg.transform(dataset.gender)
lem.fit(dataset.ever_married)
dataset['ever_married'] = lem.transform(dataset.ever_married)
lew.fit(dataset.work_type)
dataset['work_type'] = lew.transform(dataset.work_type)
ler.fit(dataset.Residence_type)
dataset['Residence_type'] = ler.transform(dataset.Residence_type)
les.fit(dataset.smoking_status)
dataset['smoking_status'] = les.transform(dataset.smoking_status)

print(dataset.head(100))

# separate x and y
x = dataset.drop('stroke', axis=1)
y = dataset['stroke']

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# train the model
model = RandomForestClassifier()
model.fit(x_train.values, y_train.values)

# score
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))

# predict
new_pred = model.predict([[1, 67, 0, 1, 1, 2, 1, 1]])
print(new_pred)
if new_pred == 1:
    print("Patient will hava stroke")
else:
    print("Patient wont hava stroke")


def predict(data):
    print(data)
    gender = leg.transform([data["gender"]]).tolist()[0]
    ever_married = int(data["ever_married"])
    work_type = lew.transform([data["work_type"]]).tolist()[0]
    Residence_type = ler.transform([data["residence_type"]]).tolist()[0]
    smoking_status = les.transform([data["smoking_status"]]).tolist()[0]
    age = int(data["age"])
    hypertension = int(data["hypertension"])
    heart_disease = int(data["heart_disease"])
    new_pred = model.predict([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, smoking_status]])
    print([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, smoking_status]])
    return new_pred.tolist()[0]


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["POST", "OPTIONS"])
def api_create_order():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        new_pred = predict(request.json)
        return _corsify_actual_response(jsonify({"prediction": new_pred}))


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response


def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

