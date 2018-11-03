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




class Detector():
    def __init__(self, graph, labels, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.graph = self.__load_graph(graph)
        self.category_index = label_map_util.create_category_index_from_labelmap(labels)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(graph=self.graph, config=self.config)




    def __load_graph(self, path_to_frozen_graph):
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def detect(self, frame):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        # Actual detection.
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.session.run([boxes, scores, classes, num_detections],
                                                            feed_dict={image_tensor: image_np_expanded})
        # Applying visual utilities to the image array (green boxes, labels, confidence score)
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=-1)

        return frame, scores

# if __name__=='__main__':
#
#     d1 = Detector(graph='models/lp_detection_graph.pb', labels='license_plate_label_map.pbtxt')
#
#     stream = cv2.VideoCapture('/home/valkov/Desktop/Video/rob/done/20181024_163259.mp4')
#     while True:
#         ret, frame = stream.read()
#         if ret:
#             frame, scores = d1.detect(frame)
#
#             cv2.imshow('frame', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     stream.release()
#     cv2.destroyAllWindows()