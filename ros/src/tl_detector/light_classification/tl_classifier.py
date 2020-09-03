from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import rospy

tl_states = {0: 'RED', 1: 'YELLOW', 2: 'GREEN', 4: 'UNKNOWN'}

class TLClassifier(object):
    def __init__(self, is_site):
        self.inference_file = ''
        if is_site:
            #self.inference_file = '../../../data/frozen_inference_graph_site.pb'
            self.inference_file = '../../../data/frozen_darknet_yolov3_model.pb'
        else:
            self.inference_file = '../../../data/frozen_inference_graph.pb'
        self.detection_graph = self.load_graph(self.inference_file)
        rospy.loginfo('Graph file loaded: %s', self.inference_file)
        self.sess = tf.Session(graph=self.detection_graph)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detect_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detect_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detect_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def classify(self, image):
        """Classifies traffic light image"""
        with self.detection_graph.as_default():              
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.detect_boxes, self.detect_scores, self.detect_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})
        return (scores, classes)

    def interpret_classification(self, scores, classes):
        """Interprets classifier output"""
        if scores is None or scores[0][0] <= 0.5:
            return TrafficLight.UNKNOWN
        else:
            if int(classes[0][0]) == 1:
                return TrafficLight.GREEN
            elif int(classes[0][0]) == 2:
                return TrafficLight.RED
            elif int(classes[0][0]) == 3:
                return TrafficLight.YELLOW
            else:
                return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        scores, classes = self.classify(image)

        res = self.interpret_classification(scores, classes)
        rospy.loginfo('TL classification result: %s', tl_states[res])

        return res
