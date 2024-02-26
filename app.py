import os
from flask import Flask, redirect, render_template, request, make_response, url_for, session
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import sklearn
import pickle
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, login_user, logout_user, UserMixin 
from flask_wtf import FlaskForm

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model_0 = CNN.CNN(39)    
model_0.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model_0.eval()

model_1 = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model_0(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///finalfinal.db'
app.config["SECRET_KEY"] = "thisisasecretkey"
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return UserLoginData.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized_callback():
    return redirect('/login')

class UserLoginData(db.Model,UserMixin):
    id = db.Column(db.Integer,primary_key = True)
    fullname = db.Column(db.String(200), nullable = True)
    username = db.Column(db.String(200), nullable = False)
    email = db.Column(db.String(200), nullable = True)
    password = db.Column(db.String(200), nullable = False)
    age = db.Column(db.Integer,nullable = True)
    

with app.app_context():
    db.create_all()


@app.route('/')
def default_page():
    return redirect('/login')


@app.route('/login',methods = ["POST","GET"])
def login_page():
    if request.method == "POST":
        username_input = request.form["username"]
        password_input = request.form["password"]
        
        user = UserLoginData.query.filter_by(username = username_input).first()
        
        if user:
            if user.password == password_input:
                login_user(user)
                return redirect('/home')

    return render_template('login.html')

@app.route('/logout', methods = ["POST","GET"])
@login_required
def logout_page():
    logout_user()
    return redirect('/login')

@app.route('/register', methods = ["POST", "GET"])
def register_page():
    if request.method == "POST":
        username_input = request.form["username"]
        password_input = request.form["password"]
        fullname_input = request.form["fullname"]
        email_input = request.form["email"]
        age_input = request.form["age"]
        
        new_user = UserLoginData(username=username_input, password=password_input,fullname =fullname_input,email = email_input,age = age_input )
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    else:
        return render_template('register.html')


@app.route('/home', methods = ["POST","GET"])
@login_required
def home_page():
    return render_template('home.html')

        
#start checking here
@app.route('/contact')
@login_required
def contact():
    return render_template("contact-us.html")


@app.route('/index')
@login_required
def ai_engine_page():
    return render_template('index.html')

@app.route('/index2')
@login_required
def index2():
    return render_template('index2.html')

@app.route('/mobile-device')
@login_required
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/predict')
@login_required
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model_1.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index2.html',result = result)

@app.route('/submit')
@login_required
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market')
@login_required
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
