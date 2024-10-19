from flask import Flask, render_template, request, redirect, url_for
from flask import request
from PIL import Image
import tensorflow as tf

app = Flask(__name__, static_folder='static', static_url_path='/static')
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/signin', methods=['POST'])
@app.route('/login', methods=['POST'])
def login_process():
    # Process Log In data here if needed
    return redirect(url_for('homepage'))

@app.route('/home')
def homepage():
    return render_template('home.html')
def signin():
    # Render the sign-in page using a template
    return render_template('signin.html') 
 
@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded file temporarily
        # Replace 'path_to_uploaded_file' with your actual path
        file_path = "C:\\dpev\\static\\images\\testimg.jpg"  
        file.save(file_path)

        # Process the uploaded image using program1.py
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        import numpy as np
        from tensorflow.keras.models import Model, load_model


        # Load the pre-trained model
        model = load_model("C:\\dpev\\model\\m1.h5")
        # Preprocess the image for prediction
        def preprocess_image(img_path):
            img = Image.open(img_path)
            img = img.resize((150, 150))  # Ensure the image is resized to match the model's input shape
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array /= 255.0  # Normalize pixel values
            return img_array
        
        # Make prediction using the loaded model
        # Assuming 'file_path' contains the path to the uploaded image
        processed_img = preprocess_image(file_path)
        prediction = model.predict(processed_img)
        # Process the prediction result and return it
        if prediction[0][0]> 0.5 :
            result = 0
            return redirect(url_for('no_hand_found'))
        else :
            result = 1
            cnn_model_path = "C:\\dpev\\model\\m2.h5"  # Replace with your CNN model path
            cnn_model = load_model(cnn_model_path)

            # Path to the test image
            file_path = "C:\\dpev\\static\\images\\testimg.jpg"  # Replace with your test image path

            # Load and preprocess the test image
            img = image.load_img(file_path, target_size=(100, 100))  # Adjust target size accordingly
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values

            # Get predictions from the CNN model
            predictions = cnn_model.predict(img_array)

            # Class Labels
            class_labels = ['a', 'aa', 'ah', 'ai', 'au', 'e', 'ee', 'i', 'ii', 'o', 'oo', 'u', 'uu']

            # Create a dictionary to map classes to numerical values
            class_to_number = {
                'a': 1,
                'aa': 2,
                'i': 3,
                'ii': 4,
                'u': 5,
                'uu': 6,
                'e': 7,
                'ee': 8,
                'ai': 9,
                'o': 10,
                'oo': 11,
                'au': 12,
                'ah': 13
            }

            # Identify the predicted class
            max_prob = 0.0
            max_prob_class = None

            for i, class_prob in enumerate(predictions[0]):
                class_label = class_labels[i]
                if class_prob > max_prob:
                    max_prob = class_prob
                    max_prob_class = class_label

            # Get the numerical value from the dictionary
            if max_prob_class in class_to_number:
                class_number = class_to_number[max_prob_class]
                print(f"The predicted class {max_prob_class} corresponds to number {class_number}")
            else:
                print("No mapping found for the predicted class")
            n= class_number
            if(n==1):
                return redirect(url_for('a'))
            elif(n==2):
                return redirect(url_for('aa'))
            elif(n==3):
                return redirect(url_for('i'))
            elif(n==4):
                return redirect(url_for('ii'))
            elif(n==5):
                return redirect(url_for('u'))
            elif(n==6):
                return redirect(url_for('uu'))
            elif(n==7):
                return redirect(url_for('e'))
            elif(n==8):
                return redirect(url_for('ee'))
            elif(n==9):
                return redirect(url_for('ai'))
            elif(n==10):
                return redirect(url_for('o'))
            elif(n==12):
                return redirect(url_for('au'))
            elif(n==13):
                return redirect(url_for('ah'))
            else:
                return redirect(url_for('oo'))
    return redirect(url_for('homepage'))

@app.route('/no_hand_found')
def no_hand_found():
    return render_template('no_hand_found.html')
@app.route('/a')
def a():
    image_name = 'a.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/aa')
def aa():
    image_name = 'aa.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/i')
def i():
    image_name = 'i.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/ii')
def ii():
    image_name = 'ii.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/u')
def u():
    image_name = 'u.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/uu')
def uu():
    image_name = 'uu.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/e')
def e():
    image_name = 'e.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/ee')
def ee():
    image_name = 'ee.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/au')
def au():
    image_name = 'au.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/ai')
def ai():
    image_name = 'ai.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/o')
def o():
    image_name = 'o.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/oo')
def oo():
    image_name = 'oo.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/ah')
def ah():
    image_name = 'ah.jpeg'  # The name of your image file
    return render_template('a.html', image_name=image_name)
@app.route('/your_route')
def your_route():
    return render_template('your_template.html')

if __name__ == '__main__':
    app.run(debug=True)
