import numpy as np
import cv2
import tensorflow as tf
import difflib

class Reader():
    def __init__(self, graph, **kwargs):
        super(Reader, self).__init__(**kwargs)
        self.graph = self.__load_graph(graph)
        self.config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=2)
        self.session = tf.Session(graph=self.graph, config=self.config)
        self.processed_set = set()

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

    # Takes image as an array
    def read(self, image):
        # We access the input and output nodes
        x = self.session.graph.get_tensor_by_name('input_image_as_bytes:0')
        prediction = self.session.graph.get_tensor_by_name('prediction:0')
        probability = self.session.graph.get_tensor_by_name('probability:0')
        # Convert image to png
        #image = cv2.imread(image, 0) # For testing
        success, encoded_img = cv2.imencode('.png', image)
        image = encoded_img.tobytes()

        # Expand Dims
        image = np.expand_dims(image, axis=0)
        # Run session on image
        (prediction_out, probability_out) = self.session.run([prediction, probability], feed_dict={
            x: image
        })

        return prediction_out.decode('utf-8'), probability_out


    # Cheks if number has been read and adds it to the set of processed if it wasn't
    def processed(self, lp_number):
        if lp_number in self.processed_set:
            return True
        elif len(difflib.get_close_matches(lp_number, self.processed_set)) > 0:
            return True
        else:
            self.processed_set.add(lp_number)
            return False
