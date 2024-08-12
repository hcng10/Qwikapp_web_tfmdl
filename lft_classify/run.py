import tensorflow as tf

import keras
import keras_cv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import time
from warnings import warn


#from img_processor import *

crop_image_path = '/cropped/' + str(os.environ["IMAGE_NAME_CROPPED"])
result_info = '/result/' + str(os.environ["RESULT"])

h5_FILENAME = "./model_mobilenet3l.h5"

resolution = (224, 224)

class_names = ['invalid', 'negative', 'positive']
print(class_names)




# Function to center and normalize the image sample-wise
def samplewise_center_std_normalize(image):
    mean = tf.reduce_mean(image, axis=[1, 2], keepdims=True)
    stddev = tf.math.reduce_std(image, axis=[1, 2], keepdims=True)
    
    centered_std_normalized_image = (image - mean) / (stddev + 1e-7) 
    
    return centered_std_normalized_image
    

# Normalize into the range of 0-255
def normalize_to_255(image):
    min_vals = tf.reduce_min(image, axis=[1, 2], keepdims=True)
    max_vals = tf.reduce_max(image, axis=[1, 2], keepdims=True)
    
    normalized_image= (image - min_vals) / (max_vals - min_vals)
    normalized_image = normalized_image * 255
    
    return normalized_image

# Function to preprocess input image
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) #/ 255.0
    image = inference_resizer({"images": tf.expand_dims(image, axis=0)})["images"]

    image = samplewise_center_std_normalize(image)
    image = normalize_to_255(image)

    return image 


# Check if the file exists
if os.path.exists(result_info):
    # Open and read the file content
    with open(result_info, "r") as file:
        content = file.read()
        
        # Check if the word 'Success' is not in the content
        if 'Success' not in content:
            warn("Cropping wasn't successful")
else:
    # File does not exist
    warn("Cannot find a cropped result")
    exit()


# Check if the cropped photos exists
if not os.path.exists(crop_image_path):
    warn("Cannot find a cropped photos")
    exit()




# Load architecture and weights from preset
classify_mdl = keras_cv.models.ImageClassifier.from_preset(
    "mobilenet_v3_large_imagenet",
    #"efficientnetv2_b0_imagenet",
    load_weights = False,
    input_shape = resolution,
    num_classes = len(class_names),
)

#classify_mdl.summary()
classify_mdl.load_weights(h5_FILENAME)


# Define the resizing layer with padding to aspect ratio
inference_resizer = keras_cv.layers.Resizing(
    resolution[0], resolution[1], 
)


# Load image
img_tensor = preprocess_image(crop_image_path)

predictions = classify_mdl.predict(img_tensor)
predicted_class = class_names[predictions[0].argmax()]

print("The predicted class is: " + str(predicted_class))

# Finally, save the test result to a text file, so it can be read back from QwikApp
with open(result_info, "w") as f:
    f.write(result_info)
    f.close()
