import config
import cv2
import numpy as np
import util

map_dimensions = None
map_corners = [[34.80, -86.71], [34.66, -86.52]]
map_name = "huntsville_map"


def load_map():
    global map_dimensions
    map = cv2.imread("raw\\" + map_name + ".jpg")
    map_dimensions = (map.shape[1], map.shape[0])
    return map


def pixel_to_lat_lon(xy):
    # This function requires the image to use a lat lon projection to be accurate
    lat = util.interpolate(
        map_corners[0][0], map_corners[1][0], xy[0] / map_dimensions[1]
    )
    lon = util.interpolate(
        map_corners[0][1], map_corners[1][1], xy[1] / map_dimensions[0]
    )
    return (lat, lon)


def lat_lon_to_pixel(lalon):
    # This function requires the image to use a lat lon projection to be accurate
    assert map_corners[0][0] > lalon[0] > map_corners[1][0]
    assert map_corners[0][1] < lalon[1] < map_corners[1][1]
    x = int(
        (map_corners[0][0] - lalon[0])
        / (map_corners[0][0] - map_corners[1][0])
        * map_dimensions[1]
    )
    y = int(
        abs(-map_corners[0][1] + lalon[1])
        / (-map_corners[0][1] + map_corners[1][1])
        * map_dimensions[0]
    )
    return np.array([x, y])


def map_images(pixel, map, size):
    opix = lat_lon_to_pixel(util.offset_lat_lon(pixel_to_lat_lon(pixel), (size, size)))
    d = abs(pixel - opix)
    y = map[
        int(pixel[0] - d[0]) : int(pixel[0] + d[0]),
        int(pixel[1] - d[1]) : int(pixel[1] + d[1]),
    ]
    resized = cv2.resize(
        y, tuple(config.image_dimensions), interpolation=cv2.INTER_AREA
    )
    return resized


def map_image(pixel, map):
    return map_images(pixel, map, config.fov)

