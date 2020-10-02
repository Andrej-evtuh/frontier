"""
File containing recogniser thread class, that is responsible for taking images from the list
and detecting faces on those images.
"""
import sys
import threading
import time
from threading import Lock
import cv2
import imutils
import numpy as np

# from utils.counter import Counter
from utils.logger import Logger
from utils import visualization_utils_color as vis_util
from connectors.config_connector import ConfigReader


class Detector:

    def __init__(self, model, max_boxes_to_draw, min_score_thresh):
        logger = Logger()
        self.log = logger.logger
        self.error_log = logger.err_logger
        self.model = model
        self.max_boxes_to_draw = max_boxes_to_draw
        self.min_score_thresh = min_score_thresh

    def detect(self, batch_frame_list):

        # resize batch_frame_list
        frames_resized = self.resize_and_stack(batch_frame_list)

        # print("RECOGNIZING")
        try:
            (boxes, scores, classes, num_detections) = self.model.sess.run(
                [self.model.boxes, self.model.scores, self.model.classes, self.model.num_detections],
                feed_dict={self.model.image_tensor: frames_resized})

            box_list = []
            face_list = []

            for i in range(len(batch_frame_list)):
                face, bbox = vis_util.get_faces_from_boxes(
                    batch_frame_list[i],
                    np.squeeze(boxes[i]),
                    np.squeeze(scores[i]),
                    max_boxes_to_draw=self.max_boxes_to_draw,
                    min_score_thresh=self.min_score_thresh
                )
                box_list.append(np.squeeze(bbox))
                face_list.append(np.squeeze(face))

            return face_list

        except:
            self.log.error("Model or face crop failure")
            self.error_log.error("Model or face crop failure", exc_info=True)

    @staticmethod
    def resize_and_stack(frames):
        tmp_list = []
        for tmp_frame in frames:
            tmp_frame = cv2.resize(tmp_frame, (300, 300))
            # tmp_frame = imutils.resize(tmp_frame, width=750)
            tmp_list.append(tmp_frame)
        frames_resized = np.stack(tmp_list)
        return frames_resized
