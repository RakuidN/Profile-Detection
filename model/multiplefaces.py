import tensorflow as tf
import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
#from tensorflow import keras
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
from PIL import Image# to read & resize the image
import keras
import tensorflow as tf
import urllib.request
from tensorflow.keras.optimizers import RMSprop

#declaring variables
size = 249 
target_size = (size,size)
crop_img = []
coords = [] #coordinates of the rectangle
male = [] # number of males in pic
female = [] # number of females in pic
predictions = []
result = []

#Load the model
model  = tf.keras.models.load_model(r"C:\Users\anees\Downloads\coc2\Gender-Classification\xception_v5_03_0.939.h5")

cartoon_model=tf.keras.models.load_model(r"C:\Users\anees\Downloads\coc2\Gender-Classification\modelusingadam.h5", compile=False)
# custom_objects={"lr": lr_track }
def main():
    st.title("Gender Classification and Fake Profile Detection")
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Predict gender from image","Predict gender from image URL", "Predict gender from camera","Catfishing Test"])

    if app_mode == "Predict gender from image":
        uploaded_file = st.file_uploader("Upload a picture of a person to predict its gender", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            st.title("Here is the picture you've uploded")


            image = Image.open(uploaded_file)
            image.save('new_image.png')
            # pixels = img_to_array(image)/255.0
            # # pixels = pixels.reshape((224, 224, 3))
            # pixels.shape
            # pixels = cv2.resize(pixels, (224,224))
            #
            # # pixels.shape
            # y=cartoon_model.predict(pixels)
            #
            # y


            img = IMAGE.load_img('new_image.png', target_size=(249, 249, 3))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            pred = model.predict(img_array)
            # pred
            if (pred) > 1e-11:
                pred = 'Person'
                st.write(pred)
                image = Image.open(uploaded_file)
                pixels = img_to_array(image)
                # create the detector, using default weights
                detector = MTCNN()
                # detect faces in the image
                faces = detector.detect_faces(pixels)
                # get faces coordinates
                croped_img, coords= get_face_coords(uploaded_file, faces)

                #output the prediction
                result, predictions= predict_img(croped_img)
                # draw rectangle on every face
                draw_rect(coords, predictions)
                st.write(result)
                st.write("Males count:",len(male))
                st.write("Females count:",len(female))
                st.write("Faces count:",len(male)+len(female))


            else:
                pred = 'Cartoon'
                st.write(pred)


    elif app_mode == "Predict gender from camera":
        #take a picture from your camera
        picture = st.camera_input("Take a picture")
        if picture is not None:



# bin=response.content
            #
            # img_array = np.frombuffer(bin, np.uint8)
            # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # img


            st.title("Here is the picture you've taken")

            image = Image.open(picture)
            
            #st.image(image)
            pixels = img_to_array(image)

            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            faces = detector.detect_faces(pixels)
            # display faces on the original image
            croped_img, coords= get_face_coords(picture, faces)
            
            result, predictions= predict_img(croped_img)#output the prediction
            # show the plot (face detected)
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
                st.write("Both the pictures are the different. User is not verified")


    elif app_mode=="Predict gender from image URL":
        url=""
        url = st.text_input('Enter Image Url here')
        # response=requests.get(url)
        # if url !="":
            # if response.status_code == 200:
            #     with open('image.jpg', 'wb') as f:
            #         f.write(response.content)
            #         'Image saved successfully!'
            # else:
            #     'Error: Failed to download image'
        # picture = Image.open('image.jpg')
        # picture
        # pixels = img_to_array(picture)
        # pixels
        # cv2.imshow(pixels)
        if url !="":
            response=requests.get(url)

            if response.status_code == 200:
                with open('image.jpg', 'wb') as f:
                    f.write(response.content)
                    # 'Image saved successfully!'
            else:
                'Error'





            # pixels = img_to_array(image)/255.0
            # # pixels = pixels.reshape((224, 224, 3))
            # pixels.shape
            # pixels = cv2.resize(pixels, (224,224))
            #
            # # pixels.shape
            # y=cartoon_model.predict(pixels)
            #
            # y


            img = IMAGE.load_img('image.jpg', target_size=(249, 249, 3))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            pred = model.predict(img_array)
            # pred
            if (pred) > 1e-11:
                pred = 'Person'
                st.header(pred)
                image = Image.open('image.jpg')
                pixels = img_to_array(image)
                # create the detector, using default weights
                detector = MTCNN()
                # detect faces in the image
                faces = detector.detect_faces(pixels)
                # get faces coordinates
                croped_img, coords= get_face_coords('img.jpg', faces)

                #output the prediction
                result, predictions= predict_img(croped_img)

                draw_rect(coords, predictions)



            else:
                pred = 'Cartoon'
                st.header(pred)



























            bin=response.content

            img_array = np.frombuffer(bin, np.uint8)
            pixels = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            image = Image.open('image.jpg')
            # image
            pixels = img_to_array(image)
            # pixels

            # pixels.save('image1.jpg')
            # picture = Image.open('image1.jpg')

            detector = MTCNN()

            faces = detector.detect_faces(pixels)

            with open('image.jpg', 'rb') as file:
                image_data = file.read()

# Create the UploadedFile object
            picture = SimpleUploadedFile('image.jpg', image_data, content_type='image/png')

            # i1 = cv2.imread(pixels)
            # i1 = Image.fromarray(pixels)
            croped_img, coords= get_face_coords(picture, faces)

            result, predictions= predict_img(croped_img)

            draw_rect(coords, predictions)
            #
            st.write(result)
            st.write("Males count:",len(male))
            st.write("Females count:",len(female))
            st.write("Faces count:",len(male)+len(female))



# draw an image with detected objects
def get_face_coords(uploaded_file, result_list):
  # load the image
  data = pyplot.imread(uploaded_file)
  # plot the image
  pyplot.imshow(data)
  for result in result_list:
    # get coordinates
    #st.write(result['confidence'])
    if result['confidence'] > 0.96:
      x1, y1, width, height = result['box']
      x2, y2 = x1 + width, y1 + height
      coords.append([x1,y1,x2,y2,width,height])
      crop_img.append(data[y1:y2,x1:x2])
  return crop_img, coords


def predict_img(croped_img):
  for crop in croped_img:     
    #preprocess Image
    img = Image.fromarray(crop, 'RGB')
    img = img.resize(target_size)
    img = img_to_array(img)
    img = img/255.0
    img=img.reshape(1, size, size, 3)

    #Prediction
    pred = model.predict(img)
    pred = pred[0][0]
    predictions.append(pred)  
    if pred>=0.5:
      male.append(1)
      result.append("Male, Confidence is {:.2f}".format(pred))
    else:
      female.append(1)
      pred = 1-pred
      result.append("Female, Confidence is {:.2f}".format(pred))
  return result, predictions


def draw_rect(coords, predictions):
    ax = pyplot.gca()
    for i,coord in enumerate(coords):
        if predictions[i] >= 0.5:
          color = 'b'
        else:
          color = 'r'
        # create the shape
        rect = Rectangle((coord[0], coord[1]), coord[4], coord[5], gid="one", fill=False, color=color)
        ax.annotate(i, (coord[2], coord[1]), color='w', weight='bold', fontsize=7, ha='center', va='center')
        # draw the box
        ax.add_patch(rect)
    pyplot.axis('off')
    st.pyplot()

#error handler

st.set_option('deprecation.showPyplotGlobalUse', False)

if __name__=='__main__':
    main()
