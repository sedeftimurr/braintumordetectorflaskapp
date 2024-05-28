from flask import Flask, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)

# Eğitilmiş modelin yüklenmesi
model = load_model('model_4.h5')

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Sonucun alınması
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Gelen dosyanın kaydedilmesi
        f = request.files['file']
        f.save(secure_filename(f.filename))
        
        # Resmin okunması ve boyutlandırılması
        img = image.load_img(f.filename, target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        # Tahminin yapılması
        result = model.predict(img)
        
        # Sonucun dönüşü
        if result[0][0] == 1:
            prediction = ' Tümör Var'
        else:
            prediction = 'Tümör Yok'
        
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

#create env
# pip install flask, tensorflow, keras, pillow
# python -m flask run
