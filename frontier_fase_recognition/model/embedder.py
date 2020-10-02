import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import facenet


class Model:
    """
    Class holds all information about loaded facenet detection model and associated session.
    Instance of the class is created one and passed on across program
    """

    model = None
    sess = None  # tf session to work with
    path = None
    images_placeholder = None

    embeddings = None
    phase_train_placeholder = None
    embedding_size = None

    def __init__(self, model_addr):
        """
        Creates graph, session instance and loads the model
        :param model_addr: Path to model
        """
        self.sess = tf.Session()
        facenet.load_model(model_addr)

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]


class Embeddings:

    def __init__(self, address):
        """
        Initializes the instance of Model class
        :param address: path to model
        """
        self.model = Model(address)

    def getEmbedding(self, resized):
        """
        Creates face embeddings of photos received in incoming parameter
        :param resized: list of face image matrices 160x160x3
        :return: list of face hashes of the same length as resized
        """
        embedding = self.model.sess.run(self.model.embeddings,
                                        feed_dict={self.model.images_placeholder: resized,
                                                   self.model.phase_train_placeholder: False})
        return embedding
