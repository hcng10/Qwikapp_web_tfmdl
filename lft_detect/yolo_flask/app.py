import tensorflow as tf

import keras
import keras_cv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import time
from warnings import warn

from flask import Flask, request, jsonify

from img_processor import *

app = Flask(__name__)


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

#image_path = os.environ["IMAGE_NAME"]
#crop_image_path = os.environ["IMAGE_NAME_CROPPED"]
#result_info = os.environ["RESULT"]

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


# Run inference
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image paths from the request
        image_path = request.json['image_path']
        crop_image_path = request.json['crop_image_path']
        result_info = request.json['result_info']
        

        # Check if files are accessible
        if not os.path.isfile(image_path):
            return jsonify({"error": f"Image file {image_path} not found"}), 400
        if not os.path.isdir(os.path.dirname(crop_image_path)):
            return jsonify({"error": f"Directory for cropped image {crop_image_path} not found"}), 400
        if not os.path.isdir(os.path.dirname(result_info)):
            return jsonify({"error": f"Directory for result info {result_info} not found"}), 400

        
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


        if is_cropped:
            print(f"Cropped image saved to: {crop_image_path}")
            with open(result_info, "w") as f:
                f.write("Cropped image saved successfully")
        else:
            warn("A test area could not be found. Returning the original image!")
            with open(result_info, "w") as f:
                f.write("Cannot locate LFT")

        return jsonify({"status": "success", "crop_image_path": crop_image_path})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)