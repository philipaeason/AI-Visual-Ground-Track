import config
import cv2
import math
import numpy as np
import random
import satellite_map
import sys
import tour
import tracker
import util


def main():
    command = "train"
    if len(sys.argv) == 2:
        command = sys.argv[1]
    if command == "show":
        show_data(range(0, config.available_count))
    if command == "kml":
        kml()
    if command == "process":
        process_training_data()
    if command == "train":
        tracker.tracker.train()
    if command == "track":
        tracker.tracker.track(0)


def kml():
    new_tour = tour.tour()
    new_tour.make_tour()
    new_tour.make_kml()


def process_training_data():
    # Create training data from video a list of viewpoints
    map = satellite_map.load_map()
    for num in range(0, config.available_count):
        tour_data = util.load("raw\\" + str(num) + "_tour")
        video_data = np.zeros((config.datafile_len, 240, 240, 3), dtype="uint8")
        viewpoint_list = []

        frame_rate = 30
        frame_time = tour_data.step / (tour_data.interp_points + 1)
        frame_stride = 30 * frame_time
        frame_no = 0

        cap = cv2.VideoCapture("raw\\" + str(num) + "_video.mp4")
        if cap.isOpened() == False:
            print("Error opening video stream or file")
            exit(1)
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        duration_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        accumulator = frame_stride
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

        burn_count = 6  # There are six extra frames at the beginning of each video
        for i in range(0, burn_count):
            ret, frame = cap.read()
        rate_correction = (1000 * tour_data.tour_time) / (
            duration_msec - 1000 * burn_count / 30
        )  # The video is not exactly 30 frames per second

        used_frame_count = 0
        while used_frame_count < config.datafile_len:
            ret, frame = cap.read()
            if ret == True:
                accumulator += rate_correction
            if accumulator >= frame_stride:
                accumulator -= frame_stride
                shape = frame.shape
                cut = int((shape[1] - shape[0]) / 2)
                frame = frame[0 : shape[0], cut : shape[1] - cut]
                frame = cv2.resize(
                    frame,
                    (config.image_dimensions[0], config.image_dimensions[1]),
                    interpolation=cv2.INTER_AREA,
                )
                video_data[used_frame_count, :, :, :] = frame
                used_frame_count += 1
            frame_no += 1
            if ret == False:
                break
        util.save("data\\" + str(num) + "_video", video_data)
        util.save("data\\" + str(num) + "_viewpoints", tour_data.viewpoints)


def show_data(range):
    map = satellite_map.load_map()
    for num in range:
        util.show(
            map,
            util.load("data\\" + str(num) + "_video"),
            util.load("data\\" + str(num) + "_viewpoints"),
        )


if __name__ == "__main__":
    main()
