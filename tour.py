import config
from datetime import datetime
from geographiclib.geodesic import Geodesic
import geopy.distance
import json
import math
import numpy as np
import random
import os
import pickle
import satellite_map
import util


class viewpoint:
    """Class for holding one point in a Google Earth tour"""

    def __init__(self, lat, lon, alt, head, i=0):
        self.head = head
        self.lat_lon = np.array([lat, lon])
        self.alt = alt


class tour:
    """Class for creating Google Earth tours"""

    def __init__(self,):
        self.alt = 200
        self.speed = 80
        self.heading = 0
        self.step = 3
        self.interp_points = 14 # The number of interpolated points to add for every step
        self.tour_len = 101
        self.tour_time = (self.tour_len - 1) * self.step
        self.upper_left_turn_back = [34.750, -86.6566] # Defines the boundary at which the tour will turn around to stay in the training area
        self.lower_right_turn_back = [34.719, -86.562]
        self.viewpoints = []

    def make_tour(self):
        # Create viewpoints that can be used for a Google Earth Tour to generate video with.
        tour_id = -1
        if os.path.isfile("tour_id.pickle"):
            tour_id = util.load("tour_id")  # Continue the IDs where they left off
        tour_id += 1
        random.seed(datetime.now())
        center = [
            util.interpolate(
                self.upper_left_turn_back[0], self.lower_right_turn_back[0], 0.5
            ),
            util.interpolate(
                self.upper_left_turn_back[1], self.lower_right_turn_back[1], 0.5
            ),
        ]
        pos = [ # Pick a random starting position
            util.interpolate(
                self.upper_left_turn_back[0],
                self.lower_right_turn_back[0],
                random.uniform(0, 1),
            ),
            util.interpolate(
                self.upper_left_turn_back[1],
                self.lower_right_turn_back[1],
                random.uniform(0, 1),
            ),
        ]
        heading = 0
        for i in range(0, self.tour_len):
            heading += random.uniform(-5, 5)
            turn_back = 0

            # Check if the tour is still in bounds
            if not (
                self.upper_left_turn_back[0] > pos[0] > self.lower_right_turn_back[0]
            ):
                turn_back = 1
            if not (
                self.upper_left_turn_back[1] < pos[1] < self.lower_right_turn_back[1]
            ):
                turn_back = 1

            if turn_back:  # Turn back towards the center of the allowed area
                head = Geodesic.WGS84.Inverse(pos[0], pos[1], center[0], center[1]).get(
                    "azi1"
                )
                angle = util.angle_between(head, heading)
                if angle > 0:
                    heading += 10
                else:
                    heading -= 10
            if heading > 180:
                heading -= 360
            if heading < -180:
                heading += 360

            origin = geopy.Point(pos[0], pos[1])
            destination = geopy.distance.distance(
                meters=self.step * self.speed
            ).destination(origin, heading)
            pos[0], pos[1] = destination.latitude, destination.longitude
            self.viewpoints.append(viewpoint(pos[0], pos[1], self.alt, heading))
        for i in range(
            0, self.tour_len - 1
        ):  # Add linearly interpolated viewpoints between the existing ones
            base = i * (self.interp_points + 1)
            base_viewpoint = self.viewpoints[base]
            next_viewpoint = self.viewpoints[base + 1]
            for i in range(0, self.interp_points):
                alpha = (i + 1) / (self.interp_points + 1)
                angle = util.angle_between(next_viewpoint.head, base_viewpoint.head)
                interpolated_viewpoint = viewpoint(
                    util.interpolate(
                        base_viewpoint.lat_lon[0], next_viewpoint.lat_lon[0], alpha
                    ),
                    util.interpolate(
                        base_viewpoint.lat_lon[1], next_viewpoint.lat_lon[1], alpha
                    ),
                    util.interpolate(base_viewpoint.alt, next_viewpoint.alt, alpha),
                    base_viewpoint.head + angle * alpha,
                    1,
                )
                self.viewpoints.insert(base + 1 + i, interpolated_viewpoint)
        util.save("raw\\" + str(tour_id) + "_tour", self)
        util.save("tour_id", tour_id)

    def make_kml(self):
        # Generate KML for a Google Earth tour
        tour_id = util.load("tour_id")
        f = open("raw\\" + str(tour_id) + "_tour.kml", "w+")

        f.write(
            """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
        <kml xmlns=\"http://www.opengis.net/kml/2.2\"
            xmlns:gx=\"http://www.google.com/kml/ext/2.2\">
            <Document>
                <name>"""
            + str(tour_id)
            + """</name>
                <open>1</open>
                <gx:Tour>
                    <name>Play</name>
                    <gx:Playlist>"""
        )
        for i in range(0, len(self.viewpoints)):
            c = self.viewpoints[i]
            f.write(
                """<gx:FlyTo>
                <gx:duration>"""
                + str(self.step / (self.interp_points + 1))
                + """</gx:duration>
		                <gx:flyToMode>smooth</gx:flyToMode>
                <Camera>
                    <longitude>"""
               + str(c.lat_lon[1])
               + """</longitude>
                    <latitude>"""
               + str(c.lat_lon[0])
               + """</latitude>
                    <altitude>"""
               + str(200)
               + """</altitude>     
                    <altitudeMode>relativeToGround</altitudeMode>
	                <heading>"""
               + str(0)
               + """</heading>        
                    <tilt>0</tilt>
                    <roll>0</roll>
                </Camera>
            </gx:FlyTo>
            """
            )

        f.write(
            """</gx:Playlist>
                </gx:Tour>
            </Document>
        </kml>"""
        )
