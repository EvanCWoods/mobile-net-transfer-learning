# import libraries
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import random

# Define the image shape and color channels
IMAGE_SHAPE = (224,224)
C_CHANNELS = 3
i = random.randint(0,1)
names = ['goldfish', 'bald_eagle']
name = names[i]
path = f'/images/{name}.jpg'

# Create the model
classifier = tf.keras.Sequential([
  hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(C_CHANNELS,))
])

# Show the image
def show_new_image(name, path):
  name = Image.open(path).resize(IMAGE_SHAPE)
  return name

show_new_image(name=name, path=path)

# Preprocess the new image
def preprocess_new_image(name):
  name = np.array(name) / 255
  print('Min: ', name.min())
  print('Max: ', name.max())
 
preprocess_new_image(name=name)

# Predict on the new image
def predict(model, name):
  result = model.predict(name[np.newaxis, ...])

  predict_label_index = np.argmax(result)
  print(predict_label_index)

  image_labels = []
  with open('labels.txt', 'r') as f:
    image_labels = f.read().splitlines()

  real_pred = image_labels[predicted_label_index]
  print(real_pred)

predict(model=classifier, name=name)

