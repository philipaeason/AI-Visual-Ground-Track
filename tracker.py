import concurrent.futures
import config
import cv2
import gc
import math
import numpy as np
import os
import pickle
from PIL import Image, ImageDraw
from pyproj import Proj, transform
import pylab
import sys
import tensorflow as tf
import random
import satellite_map
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Concatenate, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from threading import Thread, Lock
import time
import tour
import util


class tracker(object):
    """Functions to create, train, and use a convolutional neural network for the tracker"""

    randmutex = Lock()  # Prevents races with the numpy random number generator

    def shuffle(array, rng):
        tracker.randmutex.acquire()
        np.random.set_state(rng)
        np.random.shuffle(array)
        tracker.randmutex.release()

    def augment_training_data(metadata, map):
        # Creates new random misalignments between the video frames and satellite image samples.
        # The labels are the magnitudes of the misalignments
        # Samples in which there is no misalignment have a label of 1 which decreases to zero for samples with a misalignment of greater than 35 pixels
        # Misalignments range between 0 and 240 pixels with a normal distribution.
        offset_list = []
        satellite_data = np.zeros((len(metadata), 240, 240, 3), dtype="float16")
        x = 0
        while x < len(metadata):
            tracker.randmutex.acquire()
            viewpoint, angle = metadata[x]
            offset_vec = util.rotate_vec(np.array([0.0, 1.0]), random.uniform(0, 360))
            cutoff = 36
            offset_magnitude = min(abs(np.random.normal(0, 0.8 * cutoff)), 240)
            offset_vec *= offset_magnitude
            pixel = satellite_map.lat_lon_to_pixel(viewpoint.lat_lon)
            offset_pixel = np.rint(offset_vec).astype("int32")
            offset = (
                max(cutoff - np.linalg.norm(offset_pixel), 0) / cutoff
            )  # Use a normal distribution to favor small offsets because that is where accuracy is most important
            pixel += offset_pixel
            offset_list.append(offset)

            tracker.randmutex.release()
            satellite_data[x] = util.normalize_image(
                util.rotate_image(satellite_map.map_image(pixel, map), angle)
            )
            x += 1
        offset = np.array(offset_list)
        return (satellite_data, offset)

    def load_training_data(map):
        last_load = util.load("last_load")
        video_data = np.zeros((config.subset_len, 240, 240, 3), dtype="float16")
        metadata = (
            []
        )  # The metadata stores the viewpoint and random rotation associated with every video frame
        for i in range(0, config.datafiles_to_load):
            set = (i + last_load) % config.available_count
            start = config.datafile_len * i
            stop = start + config.datafile_len
            video_data[start:stop, :, :, :] = np.array(
                util.load("data\\" + str(set) + "_video")
            )
            viewpoints = np.array(util.load("data\\" + str(set) + "_viewpoints"))
            i = 0
            while i < config.datafile_len:
                ang = 90.0 * random.randint(0, 3)
                video_data[start + i] = util.normalize_image(
                    util.rotate_image(video_data[start + i].astype("uint8"), ang)
                )
                metadata.append((viewpoints[i], ang))
                i += 1
        last_load += config.datafiles_to_load
        util.save("last_load", last_load)
        validation_start = config.subset_len - config.validation_len

        satellite, offset = tracker.augment_training_data(metadata, map)
        v_validation = video_data[validation_start : config.subset_len]
        sat_validation = satellite[validation_start : config.subset_len]
        offset_validation = offset[validation_start : config.subset_len]
        video = video_data[0:validation_start]
        satellite = satellite[0:validation_start]
        offset = offset[0:validation_start]
        metadata = metadata[0:validation_start]
        state = np.random.get_state()  # Shuffle the training data
        tracker.shuffle(video, state)
        tracker.shuffle(satellite, state)
        tracker.shuffle(offset, state)
        tracker.shuffle(metadata, state)
        return (
            video,
            satellite,
            offset,
            v_validation,
            sat_validation,
            offset_validation,
            metadata,
        )

    def build_conv_layer(x, filters, strides, kernel_size):
        x = Conv2D(
            filters,
            activation="relu",
            kernel_size=kernel_size,
            strides=strides,
            padding="SAME",
            use_bias=True,
        )(x)
        return x

    def build_conv_net(scale, dim):
        # Most of the layers use four by four filters with a stride of two
        # The input images are concatenated so that the input tensor has 6 color channels, three from each image
        video_input = Input(shape=(dim[0], dim[1], 3))
        satellite_input = Input(shape=(dim[0], dim[1], 3))
        s = tf.keras.layers.Concatenate(axis=3)([video_input, satellite_input])
        x = tracker.build_conv_layer(s, int(128 * scale), 1, 3)
        x = tracker.build_conv_layer(x, int(256 * scale), 2, 4)

        for i in range(0, 4):
            x = tracker.build_conv_layer(x, int(512 * scale), 2, 4)
        x = tracker.build_conv_layer(x, int(256 * scale), 1, 3)
        x = tracker.build_conv_layer(x, int(128 * scale), 1, 1)
        x = Flatten()(x)
        return (satellite_input, video_input, x)

    def build_model(scale=0.5):
        satellite_input, video_input, conv_out = tracker.build_conv_net(
            scale, (config.image_dimensions[0], config.image_dimensions[1], 6)
        )

        x = Dense(int(2048 * scale), activation="relu", kernel_initializer="normal")(
            conv_out
        )
        x = Dense(int(1024 * scale), activation="relu", kernel_initializer="normal")(x)
        x = Dense(int(512 * scale), activation="relu", kernel_initializer="normal")(x)
        output = Dense(1, kernel_initializer="normal")(x)
        model = Model(inputs=[satellite_input, video_input], outputs=[output])
        return model

    def show_training_data(video_data, satellite_data, offset):
        for x in range(0, len(video_data), 1):
            cv2.imshow("vid", util.unnormalize_image(video_data[x]))
            cv2.imshow("sat", util.unnormalize_image(satellite_data[x]))
            print(offset[x])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def train():
        model = None
        if os.path.isfile(
            "./" + config.model_name + ".model"
        ):  # Load a model if one is the available
            model = tf.keras.models.load_model(config.model_name + ".model")
        else:
            model = tracker.build_model()
            model.summary()

        # The Huber loss helps prevent overshooting for very small misalignments but doesn't overemphasize harder large misalignments like mean squared error.
        loss = tf.keras.losses.Huber(delta=0.1)
        model.compile(loss=loss, optimizer=config.opt, metrics=["MAE"])
        model.save(config.model_name + ".model")

        map = satellite_map.load_map()
        executor = concurrent.futures.ThreadPoolExecutor()
        future_data = executor.submit(
            tracker.load_training_data, map
        )  # Load training data in a background thread
        while True:
            for j in range(0, 3):  # Number of iterations to run before saving the model
                gc.collect()
                (
                    video,
                    satellite,
                    offset,
                    video_validation,
                    satellite_validation,
                    offset_validation,
                    metadata,
                ) = future_data.result()
                future_data = executor.submit(tracker.load_training_data, map)
                future_augment = executor.submit(
                    tracker.augment_training_data, metadata, map
                )  # Augment the training data in a background thread
                last = 2
                for k in range(0, last):
                    model.fit(
                        [satellite, video],
                        [offset],
                        validation_data=(
                            [satellite_validation, video_validation],
                            [offset_validation],
                        ),
                        epochs=1,
                        batch_size=config.batch_size,
                    )
                    satellite, offset = future_augment.result()
                    if k != last - 1:
                        future_augment = executor.submit(
                            tracker.augment_training_data, metadata, map
                        )
            model.save(config.model_name + ".model")
            print("save")

    def track(datafile):
        # Tracks video by searching for good alignment with satellite imagery

        # Record videos of the tracker input and the corresponding satellite images it found
        input_video = cv2.VideoWriter("input_video.mp4", -1, 5, (240, 240))
        track_video = cv2.VideoWriter("track_video.mp4", -1, 5, (1024, 1024))
        map = satellite_map.load_map()
        model = tf.keras.models.load_model(config.model_name + ".model")
        video_data = np.array(util.load("data\\" + str(datafile) + "_video"))
        viewpoints = np.array(util.load("data\\" + str(datafile) + "_viewpoints"))
        pos = np.array(satellite_map.lat_lon_to_pixel(viewpoints[0].lat_lon))

        def ordering_func(e):
            return 1 - e[1]

        x = 1
        while x < len(viewpoints):
            viewpoint = viewpoints[x - 1]
            vid = video_data[x]
            velocity = util.rotate_vec(
                np.array([1.0, 0.0]), viewpoint.head + random.uniform(-11, 2)
            )
            velocity[0] = -velocity[0]
            velocity *= 27 + random.uniform(-4, 6)
            dead_reckoning_pos = np.rint(
                pos + velocity
            )  # Use a noisy simulated dead reckoning to help the tracker in situations where it can't find a good solution

            beam = 8  # Width of beam search
            prediction = model.predict(
                [
                    [
                        util.normalize_image(
                            satellite_map.map_image(dead_reckoning_pos, map)
                        )
                    ],
                    [util.normalize_image(vid)],
                ]
            )
            best = (dead_reckoning_pos, prediction)
            candidates = [best]
            searched_pos = []  # Don't search the same position twice
            i = 0
            while i < 8:  # Run the search for 8 steps
                j = 0
                while j < beam and j < len(candidates):
                    candidates.sort(key=ordering_func)
                    candidate = candidates[0]
                    candidates.remove(candidate)
                    search_distance = int(36 * (1 - candidate[1]) / 3)
                    if (
                        candidate[1] < 0.1
                    ):  # Search further when the predicted alignment is very bad
                        search_distance = 50
                    for search_offset in [
                        (search_distance, 0),
                        (0, search_distance),
                        (-search_distance, 0),
                        (0, -search_distance),
                    ]:
                        search_pos = candidate[0] + search_offset
                        if util.array_almost_equal(search_pos, searched_pos):
                            continue
                        searched_pos.append(search_pos)
                        prediction = model.predict(
                            [
                                [
                                    util.normalize_image(
                                        satellite_map.map_image(search_pos, map)
                                    )
                                ],
                                [util.normalize_image(vid)],
                            ]
                        )
                        new_candidate = (search_pos, prediction)
                        if prediction > best[1]:
                            best = new_candidate
                        candidates.append(new_candidate)
                    j += 1
                i += 1
            pos = best[0]
            print("True pixel:")
            print(satellite_map.lat_lon_to_pixel(viewpoint.lat_lon))
            print("Predicted pixel:")
            print(pos)
            print("Dead reckoning velocity:")
            print(velocity)
            print("")
            input_video.write(np.uint8(vid))
            track_video.write(
                np.uint8(
                    cv2.resize(
                        satellite_map.map_images(pos, map, config.fov),
                        (1024, 1024),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            )
            cv2.imshow("Video", vid)
            cv2.imshow("Solution", satellite_map.map_images(pos, map, config.fov))
            cv2.waitKey(0)
            x += 1
        cv2.destroyAllWindows()
        track_video.release()
        input_video.release()

