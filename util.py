import cv2
import math
import numpy as np
from numpy import array, array_equal, allclose
import pickle
import satellite_map


def interpolate(x, y, alpha):
    return x * (1 - alpha) + y * alpha


def angle_between(target, origin):
    # Computes the shortest angle between two angles
    angle = target - origin
    if angle > 180:
        angle -= 360
    if angle < -180:
        angle += 360
    return angle


def offset_lat_lon(lat_lon, meter_offset):
    # Offsets a latitude and longitude by meters
    lat = lat_lon[0] + meter_offset[0] / 111111
    lon = lat_lon[1] + meter_offset[1] / (111111 * np.cos(math.radians(lat_lon[0])))
    return np.array([lat, lon])


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def normalize_image(input_image):
    return (input_image / 127.5) - 1


def unnormalize_image(input_image):
    array = (input_image + 1) * 127.5
    return array.astype("uint8")


def rotate_vec(yx, degrees):
    y, x = yx
    c, s = np.cos(math.radians(degrees)), np.sin(math.radians(degrees))
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    return np.array((float(m.T[1]), float(m.T[0])))


# test for approximate equality
def array_almost_equal(myarr, list_arrays):
    return next(
        (
            True
            for elem in list_arrays
            if elem.size == myarr.size and allclose(elem, myarr)
        ),
        False,
    )


def save(name, object):
    with open(name + ".pickle", "wb") as filehandle:
        pickle.dump(object, filehandle, protocol=pickle.HIGHEST_PROTOCOL)
        filehandle.close()


def load(name):
    with open(name + ".pickle", "rb") as filehandle:
        data = pickle.load(filehandle)
        filehandle.close()
        return data


def show(map, video_data, viewpoint_list):
    # Displays video and satellite images for verifying that training data has been generated correctly
    for x in range(0, len(video_data), 299):
        viewpoint = viewpoint_list[x]
        map_view = satellite_map.map_image(
            satellite_map.lat_lon_to_pixel(viewpoint.lat_lon), map
        )
        vid = video_data[x]
        cv2.imshow("vid", vid)
        cv2.imshow("sat", map_view)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
