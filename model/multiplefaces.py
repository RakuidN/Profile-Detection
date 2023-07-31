import tensorflow as tf
import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from deepface import DeepFace
import cv2
import io
import requests

from io import BytesIO

from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from tensorflow.keras.preprocessing import image as IMAGE
from PIL import Image
import keras
import tensorflow as tf
import urllib.request
from tensorflow.keras.optimizers import RMSprop

# Declaring variables
size = 249 
target_size = (size,size)
crop_img = []
coords = []
male = []
female = []
predictions = []
result = []

# Models stored on device
model  = tf.keras.models.load_model(r"C:\Users\anees\Downloads\coc2\Gender-Classification\xception_v5_03_0.939.h5")
cartoon_model=tf.keras.models.load_model(r"C:\Users\anees\Downloads\coc2\Gender-Classification\modelusingadam.h5", compile=False)

def main():
    #We are using streamlit library which basically helps us in creating frontend by using pre-created resources
    st.title("Gender Classification and Fake Profile Detection")
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Predict gender from image","Predict gender from image URL", "Predict gender from camera","Catfishing Test"]) #All of these are options which will appear on sidebar

    if app_mode == "Predict gender from image": #Predict gender from image is the option the user has selected and same can be said about else if statements
        uploaded_file = st.file_uploader("Upload a picture of a person to predict its gender", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            st.title("Here is the picture you've uploaded")
            image = Image.open(uploaded_file)
            image.save('new_image.png')
            img = IMAGE.load_img('new_image.png', target_size=(249, 249, 3))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            pred = model.predict(img_array)
            if pred > 1e-11:
                pred = 'Person'
                st.write(pred)
                image = Image.open(uploaded_file)
                pixels = img_to_array(image)
                detector = MTCNN()
                faces = detector.detect_faces(pixels)
                croped_img, coords= get_face_coords(uploaded_file, faces)
                result, predictions= predict_img(croped_img)
                draw_rect(coords, predictions)
                st.write(result)
                st.write("Males count:",len(male))
                st.write("Females count:",len(female))
                st.write("Faces count:",len(male)+len(female))
            else:
                pred = 'Cartoon'
                st.write(pred)

    elif app_mode == "Predict gender from camera":
        picture = st.camera_input("Take a picture")
        if picture is not None:
            st.title("Here is the picture you've taken")
            image = Image.open(picture)
            pixels = img_to_array(image)
            detector = MTCNN()
            faces = detector.detect_faces(pixels)
            croped_img, coords= get_face_coords(picture, faces)
            result, predictions= predict_img(croped_img)
            draw_rect(coords, predictions)
            st.write(result)
            st.write("Males count:",len(male))
            st.write("Females count:",len(female))
            st.write("Faces count:",len(male)+len(female))

    elif app_mode=="Catfishing Test":
        uploaded_file = st.file_uploader("Upload a picture of a person to predict its gender", type=['jpg', 'jpeg', 'png'])
        picture = st.camera_input("Take a picture")
        if picture and uploaded_file is not None:
            st.title("Here are the pictures you've taken")
            uploaded_file = Image.open(uploaded_file)
            uploaded_file_array = np.array(uploaded_file)
            picture = Image.open(picture)
            picture_array = np.array(picture)
            result=DeepFace.verify(picture_array,uploaded_file_array)
            if result["verified"]:
                st.write("Both the pictures are the same. User is verified")
            else:
                st.write("Both the pictures are different. User is not verified")

    elif app_mode=="Predict gender from image URL":
        url = st.text_input('Enter Image URL here')
        if url != "":
            response = requests.get(url)
            if response.status_code == 200:
                with open('image.jpg', 'wb') as f:
                    f.write(response.content)
            img = IMAGE.load_img('image.jpg', target_size=(249, 249, 3))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            pred = model.predict(img_array)
            if pred > 1e-11:
                pred = 'Person'
                st.header(pred)
                image = Image.open('image.jpg')
                pixels = img_to_array(image)
                detector = MTCNN()
                faces = detector.detect_faces(pixels)
                croped_img, coords= get_face_coords('img.jpg', faces)
                result, predictions= predict_img(croped_img)
                draw_rect(coords, predictions)
            else:
                pred = 'Cartoon'
                st.header(pred)

def get_face_coords(uploaded_file, result_list):
    data = pyplot.imread(uploaded_file)
    pyplot.imshow(data)
    for result in result_list:
        if result['confidence'] > 0.96:
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height
            coords.append([x1, y1, x2, y2, width, height])
            crop_img.append(data[y1:y2, x1:x2])
    return crop_img, coords

def predict_img(croped_img):
    for crop in croped_img:
        img = Image.fromarray(crop, 'RGB')
        img = img.resize(target_size)
        img = img_to_array(img)
        img = img/255.0
        img = img.reshape(1, size, size, 3)
        pred = model.predict(img)
        pred = pred[0][0]
        predictions.append(pred)
        if pred >= 0.5:
            male.append(1)
            result.append("Male, Confidence is {:.2f}".format(pred))
        else:
            female.append(1)
            pred = 1 - pred
            result.append("Female, Confidence is {:.2f}".format(pred))
    return result, predictions

def draw_rect(coords, predictions):
    ax = pyplot.gca()
    for i, coord in enumerate(coords):
        if predictions[i] >= 0.5:
            color = 'b'
        else:
            color = 'r'
        rect = Rectangle((coord[0], coord[1]), coord[4], coord[5], gid="one", fill=False, color=color)
        ax.annotate(i, (coord[2], coord[1]), color='w', weight='bold', fontsize=7, ha='center', va='center')
        ax.add_patch(rect)
    pyplot.axis('off')
    st.pyplot()

if __name__ == '__main__':
    main()
