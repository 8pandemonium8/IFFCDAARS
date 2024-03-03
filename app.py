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
from flask_login import LoginManager, login_required, login_user, logout_user, UserMixin, current_user
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
    
class UserCRSHistory(db.Model):
    __tablename__ = "user_crs_history_final"
    id = db.Column(db.Integer ,primary_key = True,nullable = False)
    user_id = db.Column(db.Integer ,nullable = False)
    N = db.Column(db.Float,nullable = False)
    P = db.Column(db.Float,nullable = False)
    K = db.Column(db.Float,nullable = False)
    temp = db.Column(db.Float,nullable = False)
    humidity = db.Column(db.Float,nullable = False)
    ph = db.Column(db.Float,nullable = False)
    rainfall = db.Column(db.Float,nullable = False)
    prediction = db.Column(db.String(200), nullable = False)

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

@app.route('/predict', methods = ["POST","GET"])
@login_required
def predict():
    N_input = request.form['Nitrogen']
    P_input = request.form['Phosporus']
    K_input= request.form['Potassium']
    temp_input = request.form['Temperature']
    humidity_input = request.form['Humidity']
    ph_input = request.form['Ph']
    rainfall_input = request.form['Rainfall']
    user_id_input = current_user.id
    feature_list = [N_input, P_input, K_input, temp_input, humidity_input, ph_input, rainfall_input]
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
        new_entry = UserCRSHistory( user_id = user_id_input,N = N_input,K = K_input,P = P_input,temp = temp_input ,humidity = humidity_input ,ph =  ph_input , rainfall = rainfall_input,prediction = crop)
        db.session.add(new_entry)
        db.session.commit()
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


@app.route('/profile',methods = ['GET','POST'])
@login_required
def profile():
    if current_user.is_authenticated:
        # User is logged in, access details
        details = {
            'username': current_user.username,
            'userid' : current_user.id,
            'fullname' : current_user.fullname,
            'age' : current_user.age
            
        }
    
        history = UserCRSHistory.query.filter_by(user_id = current_user.id).order_by(UserCRSHistory.id.desc())
        print(history)
        return render_template('profile.html', details = details,history = history)
    else:
        # User is not logged in, handle accordingly
        return {'message': 'You are not logged in.'}

# @app.route('/current_user',methods=['GET','POST'])
# @login_required
# def current_user():
#     return load_user(current_user_id).username

if __name__ == '__main__':
    app.run(debug=True)