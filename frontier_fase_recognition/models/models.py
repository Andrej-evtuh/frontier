""""""

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import label_map_util


class Model:

    def __init__(self, path, label_path, num_classes):
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = path

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = label_path

        self.NUM_CLASSES = num_classes

        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def get_session(self):
        # print("in sess")
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        with detection_graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=False)

            config = tf.ConfigProto(gpu_options=gpu_options,
                                    device_count={'GPU': 1})  # Use to run on cpu only (still blocks GPU)
            # config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=detection_graph, config=config)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            # image_np_expanded = np.expand_dims(image_np, axis=0)
            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
