import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

dic = {0: 'labrador', 1: 'terrier', 2: 'woofwoof', 3: 'bordercollie', 3: 'bordercollie',
       4: 'bordercollie', 5: 'bordercollie', 6: 'bordercollie', 7: 'bordercollie',
	   8: 'bordercollie', 9: 'bordercollie', 10: 'bordercollie', 11: 'bordercollie', 
	   12: 'bordercollie', 13: 'bordercollie', 14: 'bordercollie', 15: 'bordercollie',
	   16: 'bordercollie', 17: 'bordercollie', 18: 'bordercollie', 19: 'bordercollie'}

model = load_model('model_0.h5')

def predict_label(img_path):
	img = image.load_img(img_path, target_size=(200, 200))
	img = image.img_to_array(img)
	img = img.reshape(1, 200, 200, 3)
	p = model.predict(img)
	print(p)
	return dic[np.argmax(p)]

# Routes
@app.route("/", methods=['GET', 'POST'])
def homepage():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "Sneha Seenuvasavarathan - https://www.linkedin.com/in/svsneha/"

@app.route("/submit", methods = ['GET', 'POST'])
def send_img():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/uploads" + img.filename
		img.save(img_path)
		p = predict_label(img_path)

	return render_template("home.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	app.run(debug = False)