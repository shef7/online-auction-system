from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')
model1 = pickle.load(open('mobile.pkl', 'rb'))
model2 = pickle.load(open('car.pkl', 'rb'))
model3 = pickle.load(open('camera.pkl', 'rb'))
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')
@app.route("/mobile", methods=['GET', 'POST'])
def mobile():
    return render_template('mobile.html')
@app.route("/car", methods=['GET', 'POST'])
def car():
    return render_template('car.html')

@app.route("/camera", methods=['GET', 'POST'])
def camera():
    return render_template('camera.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        battery_power = int(request.form.get('Battery_Power'))
        clock_speed = float(request.form.get('Clock_Speed'))
        front_camera = float(request.form.get('FC'))
        internal_memory = int(request.form.get('Int_Memory'))
        mobile_depth = float(request.form.get('Mobile_D'))
        mobile_weight = float(request.form.get('Mobile_W'))
        cores = int(request.form.get('Cores'))
        primary_camera = float(request.form.get('PC'))
        pixel_resolution_height = int(request.form.get('Pixel_H'))
        pixel_resolution_width = int(request.form.get('Pixel_W'))
        ram = int(request.form.get('Ram'))
        screen_height = float(request.form.get('Screen_H'))
        screen_width = float(request.form.get('Screen_W'))
        talk_time = int(request.form.get('Talk_Time'))
        has_4g = int(request.form.get('Four_G') == 'Yes')
        has_3g = int(request.form.get('Three_G') == 'Yes')
        has_touch_screen = int(request.form.get('Touch_Screen') == 'Yes')
        has_dual_sim = int(request.form.get('Dual_SIM') == 'Yes')
        has_bluetooth = int(request.form.get('Bluetooth') == 'Yes')
        has_wifi = int(request.form.get('WiFi') == 'Yes')

        data = np.array([[battery_power, clock_speed, front_camera, internal_memory, mobile_depth, mobile_weight,
                          cores, primary_camera, pixel_resolution_height, pixel_resolution_width, ram, screen_height,
                          screen_width, talk_time, has_4g, has_3g, has_touch_screen, has_dual_sim, has_bluetooth, has_wifi]])

        output = model1.predict(data)
        if output < 0:
            return render_template('mobile.html', prediction_text="Sorry, you cannot sell this mobile.")
        else:
            c=0
            if((output==1)):
                c=11200
            elif((output==2)):
                c=32000
            else:
                c=67490
            return render_template('mobile.html', prediction_text="You can sell the mobile at {}".format(c))
    else:
        return render_template('home.html')
    
@app.route("/predict1", methods=['POST'])
def predict1():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Owner=int(request.form['Owner'])
        Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol=='Petrol'):
                Fuel_Type_Petrol=1
                Fuel_Type_Diesel=0
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1
        Year=2020-Year
        Seller_Type_Individual=request.form['Seller_Type_Individual']
        if(Seller_Type_Individual=='Individual'):
            Seller_Type_Individual=1
        else:
            Seller_Type_Individual=0	
        Transmission_Mannual=request.form['Transmission_Mannual']
        if(Transmission_Mannual=='Mannual'):
            Transmission_Mannual=1
        else:
            Transmission_Mannual=0
        data = np.array([[Present_Price,Kms_Driven2,Owner,Year,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual]])
        output=model2.predict(data)
        if output<0:
            return render_template('car.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('car.html',prediction_text="You Can Sell The Car at {}".format(output))
        
    else:
        return render_template('home.html')
    
@app.route("/predict2", methods=['POST'])
def predict2():
     if request.method == 'POST':
          max_resolution = int(request.form.get('max_resolution'))
          low_resolution = int(request.form.get('low_resolution'))
          effective_pixels = int(request.form.get('effective_pixels'))
          zoom_wide = int(request.form.get('zoom_wide'))
          zoom_tele = int(request.form.get('zoom_tele'))
          focus_range = int(request.form.get('focus_range'))
          data = np.array([[max_resolution,low_resolution,effective_pixels,zoom_wide,zoom_tele,focus_range]])
          output=model3.predict(data)
          if output<0:
            return render_template('camera.html',prediction_texts="Sorry you cannot sell this camera")
          else:
            return render_template('camera.html',prediction_text="You Can Sell The Camera at {}".format(output))
          
     else:
        return render_template('home.html')

         



if __name__ == "__main__":
     app.run(debug=True )