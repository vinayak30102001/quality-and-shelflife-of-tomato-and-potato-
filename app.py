import os
import uuid
import flask
import urllib
from PIL import Image
import tensorflow
import numpy as np
from tensorflow import keras
from keras.models import load_model
from flask import Flask, render_template, request, send_file
from keras.utils import load_img, img_to_array

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'model_vgg.h5'))
model_sh=load_model(os.path.join(BASE_DIR, 'vgg_shelflife.h5'))

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ['POTATO BAD',
           'POTATO GOOD ',
           'TOMATO BAD',
           'TOMATO BAD Anthracnose',
           'TOMATO BAD Fruit Cracking',
           'TOMATO BAD Wilt',
           'TOMATO GOOD']
classes_sh=['0 DAYS',
            '1-5 DAYS',
            '5-10 DAYS',
            '10-15 DAYS',
            'STORE IN DRY CONDITIONS FOR LOGER USE',
            'USE IT ASAP OR THROW IT']

def predict1(filename, model):
    img = load_img(filename, target_size=(256, 256))
    img = img_to_array(img)
    img = img.reshape(1, 256, 256, 3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)
    result = np.argmax(result, axis=-1)
    #print(result)
    class_result = classes[result[0]]

    return class_result
def predict_sh(filename, model_sh):
    img = load_img(filename, target_size=(256, 256))
    img = img_to_array(img)
    img = img.reshape(1, 256, 256, 3)

    img = img.astype('float32')
    img = img/255.0
    result_sh = model_sh.predict(img)
    result_sh = np.argmax(result_sh, axis=-1)
    class_result_sh = classes_sh[result_sh[0]]

    return class_result_sh

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result= predict1(img_path, model)
                class_result_sh=predict_sh(img_path, model_sh)
               
                predictions = {
                    "class1": class_result,
                    "prob1": class_result_sh,
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result= predict1(img_path, model)
                class_result_sh=predict_sh(img_path, model_sh)
                predictions = {
                    "class1": class_result,
                    "prob1": class_result_sh,
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
