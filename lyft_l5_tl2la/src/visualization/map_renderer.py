# ------------------------------------------------------------------------
# Visualizer based on Matplotlib to render Map features of
# Lyft's HD map
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.dataset_operations.extended_map_api import ExtendedMapAPI


class StaticMapRenderer:
    """Fetch semantic attributes over the map api and visualize it"""

    def __init__(self, map_api):
        self._color_mapping = dict(drivable_area='#afd1e3',
                                    road_segment='#1871ad',
                                    road_block='#b2df8a',
                                    lane='#474747',
                                    traffic_control_element='#ff0000',
                                    traffic_light="#00db04",
                                    junctions='#fcba03')

        self._map_api: ExtendedMapAPI = map_api
        self.fig = plt.figure(figsize=(10, 10))

    def add_layer(self, name: str):
        ax = self.fig.gca()
        #print("Layer name:", name)
        return ax

    def render_lanes(self):
        lanes = self._map_api.get_elements_from_layer("lane")
        ax = self.add_layer("lanes")
        for lane in lanes:
            self.render_lane_segment(ax, lane)
        return self.fig, ax

    def render_lane_segment(self, ax, lane):
        coords = self._map_api.get_lane_coords(ExtendedMapAPI.id_as_str(lane.id))
        self.plot_coords(ax, "lane", coords, "xyz_left")
        self.plot_coords(ax, "lane", coords, "xyz_right")

    def plot_coords(self, ax, attribute_type: str, coords: np.ndarray, access_key: str):
        xs = coords[access_key][:, 0]
        ys = coords[access_key][:, 1]
        ax.plot(
            xs, ys, color=self._color_mapping[attribute_type], label=attribute_type)

    def plot_bounding_box(self, ax, attribute_type: str, bounds: np.ndarray):
        p_south_west = bounds

        width = 50
        heigth = 50
        ax.add_patch(Rectangle((p_south_west[0]-25, p_south_west[1]-25), width,
                     heigth, edgecolor="blue", facecolor=self._color_mapping[attribute_type], fill=True, label=attribute_type, zorder=10))

    def render_traffic_control_elements(self):
        traffic_elements = self._map_api.get_elements_from_layer("traffic_control_element")
        ax = self.add_layer("traffic_control_elements")
        for traffic_element in traffic_elements:
            self.render_traffic_control_element(ax, traffic_element)
        return self.fig, ax

    def render_traffic_control_element(self, ax, traffic_element):
        coords = self._map_api.get_traffic_control_element_coords(
            ExtendedMapAPI.id_as_str(traffic_element.id)
        )
        self.plot_coords(ax, "traffic_control_element", coords, "xyz")

    def render_traffic_lights(self):
        traffic_elements = self._map_api.get_elements_from_layer("traffic_control_element")

        # Filter for traffic lights
        traffic_lights = [
            elem for elem in traffic_elements if self._map_api.is_traffic_light(elem)]
        ax = self.add_layer("traffic_control_elements")
        for traffic_light in traffic_lights:
            self.render_traffic_light(ax, traffic_light)
        return self.fig, ax

    def render_traffic_light(self, ax, traffic_element):
        coords = self._map_api.get_traffic_light_coords(
            ExtendedMapAPI.id_as_str(traffic_element.id)
        )
        self.plot_coords(ax, "traffic_light", coords, "xyz")

    def render_junctions(self, with_tl_only=True):
        all_junctions = self._map_api.get_elements_from_layer("junction")
        ax = self.add_layer("junctions")

        if with_tl_only:
            all_junctions = self._map_api.filter_tl_junctions(all_junctions)

        for junction in all_junctions:
            self.render_junction(ax, junction)
        return self.fig, ax

    def render_junction(self, ax, junction):
        bounds = self._map_api.get_junction_coords(
            ExtendedMapAPI.id_as_str(junction.id))
        self.plot_bounding_box(ax, "junctions", bounds)
