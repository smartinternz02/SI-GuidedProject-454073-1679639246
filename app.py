import flask
from flask import request, render_template
import joblib

app = flask.Flask(__name__, static_url_path='')

@app.route('/', methods=['GET'])
def sendHomePage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predictPerformance():
    cylinders = float(request.form['cylinders'])
    displacement = float(request.form['displacement'])
    horsepower = float(request.form['horsepower'])
    weight = float(request.form['weight'])
    acceleration = float(request.form['acceleration'])
    modelyear = float(request.form['modelyear'])
    origin = float(request.form['origin'])

    X = [[cylinders,displacement,horsepower,
          weight,acceleration,modelyear,origin]]

    model = joblib.load('gbr_performance.pkl')

    rating = model.predict(X)[0]
    return render_template('predict.html',predict=round(rating))

if __name__ == '__main__':
    app.run()
    
