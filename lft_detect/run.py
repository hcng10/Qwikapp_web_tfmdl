import tensorflow as tf

import keras
import keras_cv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import time
from warnings import warn


from img_processor import *

import subprocess
result = subprocess.run(
        ["ls"],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )


h5_FILENAME = "./model_yolov8xs.h5"

obj_dl_resolution = [640, 640]

#deepblue_ofst_factors_cls0 = [0, 0.33, 1, 0.59]
deepblue_ofst_factors_cls0 = [0.4, 0.35, 0.6, 0.58]
deepblue_ofst_factors_cls1 = [0.4, 0.15, 0.6, 0.95]#xmin ymin xmax ymax

class_ids = [
    "cassette",
    "test_region",
    "test_strip"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids)) 

#image_path = '716original_2024-02-27-150444_DeepBlue_20ngmL-1.jpg'
#crop_image_path = '716original_2024-02-27-150444_DeepBlue_20ngmL-1_crop.jpg'

#image_path = '738original_2024-02-27-151112_DeepBlue_100ngmL-1.jpg'
#crop_image_path = '738original_2024-02-27-151112_DeepBlue_100ngmL-1_crop.jpg'

image_path = '/images/' + str(os.environ["IMAGE_NAME"])
crop_image_path = '/cropped/' + str(os.environ["IMAGE_NAME_CROPPED"])
result_info = '/result/' + str(os.environ["RESULT"])

#print(image_path)
#print(crop_image_path)

target_class = 2
fallback_class = {0, 1}



start = time.time()

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xs_backbone_coco",
    load_weights=False
)

self_yolo_model = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_ids),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=2,# as recommended by Keras
)

self_yolo_model.load_weights(h5_FILENAME)
#self_yolo_model.summary()

end = time.time()
print("Load YOLO time: " + str(end - start))

       
# Process image
start = time.time()
imp = ImgProcessor(image_path, obj_dl_resolution)
preprocessed_image = imp.get_tf_img()
end = time.time()
print("Image processing time: " + str(end - start))


# Run inference
start = time.time()
detect_outputs = self_yolo_model.predict(preprocessed_image)
end = time.time()
print("Inference time: " + str(end - start))


 # Post-process results
start = time.time()
boxes, scores, classes = imp.postprocess_output(detect_outputs,)
is_cropped = imp.crop_n_save(crop_image_path, boxes, classes, scores, target_class, fallback_class, [deepblue_ofst_factors_cls0, deepblue_ofst_factors_cls1])
end = time.time()
print("cropping time: " + str(end - start))


f = open(result_info, "w")
if is_cropped:
    print(f"Cropped image saved to: {crop_image_path}")
    f.write("Success locate LFT")
else:
    warn("A test area could not be found. Returning the original image!")
    f.write("Cannot locate LFT")
f.close()
        