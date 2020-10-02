""""""

import tensorflow as tf


class Model:

    def __init__(self, path, label_path, num_classes=1):

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = path

        self.NUM_CLASSES = num_classes

    def get_session(self):
        # print("in sess")
        with tf.io.gfile.GFile(self.PATH_TO_CKPT, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
        # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, device_count={'GPU': 0})
        # self.sess = tf.compat.v1.Session(config=config)
        self.sess = tf.compat.v1.Session()
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        self.image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.sess.graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.sess.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.sess.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.sess.graph.get_tensor_by_name('num_detections:0')
