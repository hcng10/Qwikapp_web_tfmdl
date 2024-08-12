import tensorflow as tf

import keras
import keras_cv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
from warnings import warn

class ImgProcessor:
    def __init__(self, image_path, obj_dl_resolution):
        # Define a layer for resizing images, preserving aspect ratio by padding on the RIGHT SIZE Only as needed
        inference_resizer = keras_cv.layers.Resizing(
            obj_dl_resolution[0], obj_dl_resolution[1], 
            bounding_box_format="xyxy",
            pad_to_aspect_ratio=True
        )

        # Read the image from the given file path
        self.image = tf.io.read_file(image_path)
        
        # Decode the image as a JPEG file, ensuring it has 3 color channels
        self.image = tf.image.decode_jpeg(self.image, channels=3)
        
        # Store the original shape (height and width) of the image
        self.original_shape = tf.shape(self.image)[:2]

        # Convert the image to float32 data type for further processing
        self.image = tf.cast(self.image, tf.float32) 
        # Resize the image using the defined inference resizer
        self.image = inference_resizer({"images": tf.expand_dims(self.image, axis=0)})["images"]

        # Store the target object detection resolution and image path
        self.obj_dl_resolution = obj_dl_resolution
        self.image_path = image_path
        

    def get_tf_img(self):
        # Return the preprocessed TensorFlow image tensor
        return self.image


    def display_preprocessed_image(self):
        # Remove the batch dimension and normalize the image for displaying
        preprocessed_image = tf.squeeze(self.image, axis=0)/255 
        plt.imshow(preprocessed_image)
        plt.title('Preprocessed Image')
        plt.axis('off')
        plt.show()
        

    def postprocess_output(self, detect_outputs, threshold=0.5):
         # Extract bounding boxes, confidence scores, and class labels from detection outputs
        boxes, scores, classes = detect_outputs["boxes"], detect_outputs["confidence"], detect_outputs["classes"]
        
        # Apply threshold to filter out low confidence detections
        mask = scores > threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

         # Get the original dimensions of the image
        original_height, original_width = tf.cast(self.original_shape[0], dtype=tf.float32), \
                                        tf.cast(self.original_shape[1], dtype=tf.float32)
        
        # Calculate the scaling factor used to resize the image while preserving aspect ratio
        scale = min(self.obj_dl_resolution[0] / original_height, self.obj_dl_resolution[1]  / original_width)

         # Calculate the padding added to height and width due to aspect ratio preservation
        pad_height = (self.obj_dl_resolution[0] - original_height * scale) 
        pad_width = (self.obj_dl_resolution[1] - original_width * scale) 
    
        
        # Scale the bounding boxes back to the original image dimensions
        boxes = tf.convert_to_tensor(boxes)
        boxes = tf.stack([
            boxes[:, 0]  / scale,
            boxes[:, 1]  / scale,
            boxes[:, 2]  / scale,
            boxes[:, 3]  / scale,
         ], axis=-1)

        return boxes, scores, classes


    def crop_n_save(self, output_path, boxes, classes, scores, target_class, fallback_class, fallback_ofst_ls):
        
        # Flag to indicate if a target class object was found and cropped
        return_val = False
        
        # Open the original image using PIL (Python Imaging Library)
        pil_image = Image.open(self.image_path)
        #pil_image= Image.open(image_path)

        # Filter boxes, classes, and scores for the target class
        target_indices = [i for i, cls in enumerate(classes) if int(cls.item()) == target_class]

        if not target_indices:
            # If no detections of the target class are found, filter indices for the fallback class
            fallback_indices = [i for i, cls in enumerate(classes) if int(cls.item()) in fallback_class]
            
            if not fallback_indices:
                # If no fallback class detections are found, save the original image
                pil_image.save(f"{output_path}", "JPEG")
                return return_val
                

            # Get the index of the fallback detection with the highest score
            best_fallback_index = max(fallback_indices, key=lambda i: scores[i])
            
            best_box = boxes[best_fallback_index].numpy().astype(int)
            best_fallback_cls = classes[best_fallback_index]

            print("best_fallback_cls", best_fallback_cls)
            fallback_ofst_factors = fallback_ofst_ls[best_fallback_cls]

            # Perform cropping from certain point of the horizontal position of the fallback box
            horizontal_start = best_box[0] + int(fallback_ofst_factors[0] * (best_box[2] - best_box[0]))
            horizontal_end = best_box[0] + int(fallback_ofst_factors[2] * (best_box[2] - best_box[0]))

            if horizontal_start == horizontal_end:
                horizontal_end = best_box[2]

            # Perform cropping from certain point of the vertical position of the fallback box
            vertical_start = best_box[1] + int(fallback_ofst_factors[1] * (best_box[3] - best_box[1]))
            vertical_end = best_box[1] + int(fallback_ofst_factors[3] * (best_box[3] - best_box[1]))

            if vertical_start == vertical_end:
                vertical_end = best_box[3]
    
            cropped_image = pil_image.crop((horizontal_start, vertical_start, horizontal_end, vertical_end))
            cropped_image.save(f"{output_path}", "JPEG")

            warn("Using fallback class for cropping!")
            return_val = True

        else:

            # Get the index of the detection with the highest score
            best_index = max(target_indices, key=lambda i: scores[i])
      
            # Get the best box
            best_box = boxes[best_index].numpy().astype(int)
            
            # Crop the image using the bounding box coordinates
            cropped_image = pil_image.crop((best_box[0], best_box[1], best_box[2], best_box[3]))
        
            # Save the cropped image to the specified output path
            cropped_image.save(f"{output_path}", "JPEG")
            return_val = True
    
        return return_val
