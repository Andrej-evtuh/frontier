"""
File description:
    File to read connector.ini config
Classes:
    ConfigReader - reads config
"""
import configparser
import os
import json
import sys
import numpy as np

from utils.logger import Logger

class ConfigReader:
    """
    Methods:
        init - initialises config instance
    """

    config = None  # config object

    def __init__(self):
        self.config = configparser.ConfigParser()

        # path for config
        path = os.getcwd()
        path_to_config = os.path.join(path, "./config/config.ini")

        self.config.read(path_to_config)

        logger = Logger()
        self.log = logger.logger
        self.error_log = logger.err_logger

    def get_folder_path_config(self):
        folder_path = self.ConfigSectionMap("folder_paths")["path"]
        return folder_path

    def get_vis_utils_config(self):
        max_boxes_to_draw = int(self.ConfigSectionMap("vis_utils_properties")["max_boxes_to_draw"])
        min_score_thresh = float(self.ConfigSectionMap("vis_utils_properties")["min_score_thresh"])
        return max_boxes_to_draw, min_score_thresh

    def get_model_config(self):
        model_address = self.ConfigSectionMap("modelStorage")["model"]
        model_protos = self.ConfigSectionMap("modelStorage")["protos"]
        return model_address, model_protos

    def get_facematch_model_config(self):
        """
        Gets all NN models parameters and structure for embedder.py
        :return: paths to models
        """
        model_address = self.ConfigSectionMap("modelStorage")["model_facematch"]
        return model_address

    def ConfigSectionMap(self, section):
        """
        :param section: section from ini file that is needed
        :return: all entries from this section as dictionary
        """
        dict1 = {}
        options = self.config.options(section)
        for option in options:
            try:
                dict1[option] = self.config.get(section, option)
                if dict1[option] == -1:
                    print("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1

    # def from_config_to_dataframe(self):
    #     cam_id_list = []
    #     video_folder_list = []
    #     roi_list = []
    #     try:
    #         roi_dict = json.loads(self.get_roi_config())
    #     except:
    #         self.log.error("Error reading line_coordinates from config.ini")
    #         self.error_log.error("Error reading line_coordinates from config.ini", exc_info=True)
    #         sys.exit()
    #     try:
    #         video_folder_list = json.loads(self.get_camera_paths_config())
    #     except:
    #         self.log.error("Error reading cameras_paths from config.ini")
    #         self.error_log.error("Error reading cameras_paths from config.ini", exc_info=True)
    #         sys.exit()
    #     # for cam_path in video_folder_list:
    #     for i in range(len(video_folder_list)):
    #         cam_path = video_folder_list[i]
    #         cam_id = cam_path.split('\\')[-1].split('/')[-1]
    #         cam_id_list.append(cam_id)
    #         roi_list.append(roi_dict[cam_id])
    #
    #     return cam_id_list, video_folder_list, roi_list

