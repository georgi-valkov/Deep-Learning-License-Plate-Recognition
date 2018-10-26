import numpy as np
import os
import sys
import time
import cv2
import base64

from PIL import ImageFont, ImageDraw, Image

import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'models/lp_detection_graph.pb'
PATH_TO_FROZEN_GRAPH_TEXT_READING = 'models/text_reading_graph.pb'
PATH_TO_LABELS = 'license_plate_label_map.pbtxt'

FONT = ImageFont.truetype('utils/consola.ttf', 30)

def load_graph(path_to_frozen_graph):
    #Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph
def read_text_from_image(session, image):
    # We access the input and output nodes
    x = session.graph.get_tensor_by_name('input_image_as_bytes:0')
    prediction = session.graph.get_tensor_by_name('prediction:0')
    probability = session.graph.get_tensor_by_name('probability:0')
    # Convert image to png
    success, encoded_img = cv2.imencode('.png', image)
    image = encoded_img.tobytes()
    # Expand Dims
    image = np.expand_dims(image, axis=0)
    # Run session on image
    (prediction_out, probability_out) = session.run([prediction, probability], feed_dict={
        x: image
    })
    return prediction_out, probability_out

# Detection graph
detection_graph = load_graph(PATH_TO_FROZEN_GRAPH)
# Reading graph
reading_graph = load_graph(PATH_TO_FROZEN_GRAPH_TEXT_READING)

#Loading label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)


vide_file_name = '/home/valkov/Desktop/Video/rob/done/20181024_163259.mp4'  # FUll PATH TO VIDEO HERE

stream = cv2.VideoCapture(vide_file_name)

# Text Reading Session
config = tf.ConfigProto(allow_soft_placement=True)
text_sess = tf.Session(graph=reading_graph, config=config)
sess = tf.Session(graph=detection_graph, config=config)



while True:

    ret, frame = stream.read()

    # Check if valid frame
    if ret:

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        # Actual detection.
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                            feed_dict={image_tensor: image_np_expanded})
        # Applying visual utilities to the image array (green boxes, labels, confidence score)
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=-1)

        if scores[0][0] > 0.995:
            height, width, channels = frame.shape
            # For All detected objects in the picture
            for i in range(int(num_detections[0])):
                # Bounding box coordinates
                ymin = int((boxes[0][i][0] * height))
                xmin = int((boxes[0][i][1] * width))
                ymax = int((boxes[0][i][2] * height))
                xmax = int((boxes[0][i][3] * width))
                # print(boxes[0][1])

                lp_np = frame[ymin:ymax, xmin:xmax]
                # cv2.imshow('License Plate - certainty > 99.5%', lp_np)
                prediction, probability = read_text_from_image(text_sess, lp_np)
                if probability > 0.95:

                    h, w, c = lp_np.shape




                    # Create new PIL image and draw the prediction inside
                    pil_img =Image.new("RGB", (w, h+50))
                    draw = ImageDraw.Draw(pil_img)
                    draw.text((0, h), prediction.decode('utf-8'), font=FONT)

                    lp_np = Image.fromarray(lp_np)

                    pil_img.paste(lp_np, (0, 0))

                    cv2.imshow('License with Prediction', np.array(pil_img))

                    # print('prediction - %s\t probability - %s'%(prediction, probability))

        # Show frames in a window
        cv2.imshow('License Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

stream.release()
cv2.destroyAllWindows()