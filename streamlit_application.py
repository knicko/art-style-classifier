import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import h5py

from PIL import Image, ImageOps

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax

st.set_option('deprecation.showfileUploaderEncoding', False)
st.header('Welcome to Project Art Thieves!')
st.write('Input a picture and we will try to predict which style it is from!')

def main():
  file = st.file_uploader('Upload An Image: ', type=["jpg", "png"])
  if file is not None:
    image = Image.open(file)
    figure = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    result = predict_class(image)
    st.write(result)
    st.pyplot(figure)

def predict_class(image):
  model = tf.keras.models.load_model('model16a.hdf5')
  size = (250,250)    
  image = ImageOps.fit(image, size, Image.ANTIALIAS)
  image = np.asarray(image)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255. 
  test_image = img[np.newaxis,...]
  class_names = ['Early-Renaissance', 'High-Renaissance', 
                   'Mannerism-(Late-Renaissance)', 'Northern-Renaissance',
                     'Baroque', 'Tenebrism', 'Rococo', 'Neoclassicism', 
                     'Academicism', 'Romanticism', 'Orientalism', 
                     'Realism', 'Naturalism', 'Naïve-Art-(Primitivism)',
                       'Social-Realism', 'Surrealism',
                         'Post-Impressionism', 'Symbolism', 
                         'Impressionism', 'Magic-Realism', 
                         'Abstract-Art', 'Art-Deco', 'Regionalism', 
                         'Socialist-Realism', 'Fauvism', 
                         'Art-Nouveau-(Modern)', 'Expressionism', 
                         'Kitsch', 'Neo-Romanticism', 'Abstract-Expressionism', 'Cubism', 'Color-Field-Painting', 'Hard-Edge-Painting', 'Lyrical-Abstraction', 'Art-Informel', 'Tachisme', 'Constructivism', 'Concretism', 'Neo-Expressionism', 'Op-Art', 'Pop-Art', 'Neo-Impressionism', 'Conceptual-Art', 'Minimalism', 'Post-Minimalism', 'Contemporary-Realism', 'Transavantgarde', 'Neo-Pop-Art', 'Contemporary', 'Ink-and-wash-painting', 'Ukiyo-e']
  
  
  prediction = model.predict(test_image)
  score = tf.nn.softmax(prediction[0])
  result = "The predicted style/genre of this artwork is {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
  return result

if __name__ == "__main__":
  main()