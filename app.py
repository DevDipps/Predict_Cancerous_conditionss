import fnmatch

import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow import keras

app = Flask(__name__)

dic = {0 : 'Benign', 1 : 'Malignant'}

model = load_model('IDCResNet.h5')

model.make_predict_function()

def predict_label(img_path):
	i = keras.utils.load_img(img_path, target_size=(50, 50))
	i = keras.utils.img_to_array(i) / 255.
	i = i.reshape(1, 50,50,3)
	p = model.predict(i)
	labels = np.array(p)
	labels[labels >= 0.5] = 1
	labels[labels <= 0.5] = 0

	print(labels)
	final = np.array(labels)

	if final[0][0] == 1:
		return "Benign"
	else:
		return "Malignant"

	return [p]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)