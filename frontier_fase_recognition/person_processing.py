"""
File contains
"""
import os
import cv2
from scipy.spatial import distance
from detector import Detector
import sys
import numpy as np

from utils.camera_helper import Camera
from models.models import Model
from connectors.config_connector import ConfigReader
from utils.logger import Logger
from utils.calculate_face import Calculate


class Person_processing:

    def __init__(self):

        logger = Logger()
        self.log = logger.logger
        self.error_log = logger.err_logger
        self.camera = None

        try:
            self.log.info("Reading the config...")
            self.config = ConfigReader()
            self.log.info("Config read")
        except:
            self.log.error("Error reading config.ini")
            self.error_log.error("Error reading config.ini", exc_info=True)
            sys.exit()

        try:
            self.folder_path = self.config.get_folder_path_config()
            print("folder_path: {}".format(self.folder_path))
        except:
            self.log.error("Initialisation error: the video folder path is not defined")
            self.error_log.error("Initialisation error: the video folder path is not defined", exc_info=True)
            sys.exit()

        self.log.info("Initialising face_model")
        try:
            face_model_address, face_model_protos = self.config.get_model_config()
            print(face_model_address, face_model_protos)
            self.get_face_model = Model(face_model_address, face_model_protos, num_classes=2)
            self.get_face_model.get_session()
            self.log.info("face_model initialisation completed")
        except:
            self.log.error("face_model initialisation error")
            self.error_log.error("face_model initialisation error", exc_info=True)
            sys.exit()

        self.log.info("Initialising camera")
        try:
            self.camera = Camera(self.folder_path)
            self.log.info("Camera initialised")
        except:
            self.log.error("Camera initialisation error")
            self.error_log.error("Camera initialisation error", exc_info=True)
            sys.exit()

        try:
            self.max_boxes_to_draw, self.min_score_thresh = self.config.get_vis_utils_config()
        except:
            self.max_boxes_to_draw, self.min_score_thresh = 10, 0.3

        self.log.info("Initializing detector")
        try:
            self.detector = Detector(self.get_face_model, self.max_boxes_to_draw, self.min_score_thresh)
            self.log.info("detector initialized")
        except:
            self.log.error("detector initialization failed")
            self.error_log.error("detector initialization failed", exc_info=True)
            sys.exit()

        try:
            self.log.info("Initialising embeddings models...")
            self.calculator = Calculate(self.log, self.error_log, self.config)
            self.log.info("embeddings Models initialisation completed")
        except:
            self.log.error("embeddings Models initialisation error")
            self.error_log.error("embeddings Models initialisation error", exc_info=True)
            sys.exit()

        self.batch_frame_list = []

        # todo set batch_size to config.ini ?
        self.batch_size = 2
        self.flag = True

    def get_filenames_list(self):
        """
        get filenames from  folder
        :return: string (video_file, scan, rfid)
        """
        video_name = None
        files = [f for f in os.listdir(self.folder_path) if
                 os.path.isfile(os.path.join(self.folder_path, f))]
        for file in files:
            if ".txt" in file:
                return None
            if ".avi" in file:
                video_name = file
        return video_name

    def get_all_frames(self, video_name):
        full_video_path = os.path.join(self.folder_path, video_name)
        # print("reading {}".format(full_video_path))
        try:
            self.camera.camera_capture.open(full_video_path)
            self.log.debug("reading {}".format(full_video_path))
        except:
            self.log.error("Cannot open video file {}".format(full_video_path))
            self.error_log.error("Cannot open video file {}".format(full_video_path), exc_info=True)
        # read frames to frame_list
        frame_list = []
        while True:
            frame, ret = self.camera.get_frame()
            if not ret:
                break
            # rgb_2_bgr
            frame = self.camera.rgb_to_bgr(frame)
            frame_list.append(frame)

            # print(frame.shape)
            # cv2.imshow('window_name', frame)
            # cv2.waitKey(300)
            # cv2.destroyAllWindows()

        # after reading video file, close reader to avoid PermissionError
        try:
            self.camera.camera_capture.release()
        except:
            self.log.error("Cannot close video file {}".format(full_video_path))
            self.error_log.error("Cannot close video file {}".format(full_video_path), exc_info=True)

        return frame_list

    def get_faces(self, frame_list):
        """
        Get all faces from video
        :param frame_list: list of frames
        :return:
        """
        face_list = []
        while len(frame_list) > 0:
            if len(frame_list) < self.batch_size:  # if len < n skip
                break
            else:
                # print("detect")
                batch = frame_list[:self.batch_size]
                frame_list = frame_list[self.batch_size:]
            face_batch = self.detector.detect(batch)
            face_list = face_list + face_batch
        return face_list

    def process(self):
        print('in process')
        while True:
            # read filenames from folder
            video_name = self.get_filenames_list()
            if video_name is None:
                continue
            # for video file get all frames
            try:
                frame_list = self.get_all_frames(video_name)
            except:
                self.log.error("Cannot read video file {}".format(video_name))
                self.error_log.error("Cannot read video file {}".format(video_name), exc_info=True)
                continue
            # for frames_list get face_list (detect face and corp)
            try:
                video_face_list = self.get_faces(frame_list)
            except:
                self.log.error("Cannot get face, detector error for {}".format(video_name))
                self.error_log.error("Cannot get face, detector error for {}".format(video_name), exc_info=True)
                continue
            # for frame in video_face_list:
            # print(frame.shape)
            # cv2.imshow('window_name', frame)
            # cv2.waitKey(100)
            # cv2.destroyAllWindows()

            # # # embeddings
            # read scan & rfid images
            scan_face = cv2.imread(os.path.join(self.folder_path, 'scan.bmp'))
            rfid_face = cv2.imread(os.path.join(self.folder_path, 'rfid.jpg'))
            # get video, scan & rfid hash
            video_face_hash = self.calculator.calculate_faces(video_face_list)
            scan_face_hash = self.calculator.calculate_faces(np.expand_dims(scan_face, axis=0))
            rfid_face_hash = self.calculator.calculate_faces(np.expand_dims(rfid_face, axis=0))
            # distance
            scan_dist = 1 - self.count_distance(video_face_hash, scan_face_hash)
            rfid_dist = 1 - self.count_distance(video_face_hash, rfid_face_hash)
            print('----')
            print(scan_dist, rfid_dist)
            print('----')
            # todo  проверить как будет менятся дистанция при обрезании края изображения для scan и rfid или увеличения краёв при обнаружении лиц
            self.save_results(scan_dist, rfid_dist)

    def count_distance(self, hash_1, hash_2):
        # Count euclidian distance matrix, between objects and faces in one image
        dist = distance.euclidean(hash_1, hash_2)
        return dist

    def save_results(self, scan_dist, rfid_dist):
        """
        save info about obj. recognitions to .txt file
        :return: None
        """
        save_path = os.path.join(self.folder_path, 'res.txt')
        try:
            text_file = open(save_path, "x")
        except FileExistsError:
            self.log.error("save_results: FileExistsError {}".format(save_path))
            self.error_log.error("save_results: FileExistsError {}".format(save_path), exc_info=True)
            pass
        except FileNotFoundError:
            self.log.error("save_results: FileNotFoundError {}".format(save_path))
            self.error_log.error("save_results: FileNotFoundError {}".format(save_path), exc_info=True)
            pass
        text_file.write(str(scan_dist)
                        + '\n'
                        + str(rfid_dist)
                        )
