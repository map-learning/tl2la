# ------------------------------------------------------------------------
# Lyft Lvl5 MapAPI with extended functionalities
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

from functools import lru_cache
from typing import List, Set, no_type_check, Tuple, Dict

import numpy as np
import pymap3d as pm
from typing import Tuple
import pandas as pd
from l5kit.configs.config import load_metadata
from l5kit.data import DataManager, MapAPI
from l5kit.data.map_api import TLFacesColors
from l5kit.data.proto.road_network_pb2 import GeoLocation, MapElement
from l5kit.geometry import transform_point
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds

from src.common.helper import calculate_center, load_pickled_df
from src.common.definitions import TURN
CACHE_SIZE = int(1e5)


class ExtendedMapAPI(MapAPI):
    def __init__(
        self,
        protobuf_map_path: str,
        world_to_ecef: np.ndarray,
        cfg
    ):
        super().__init__(protobuf_map_path, world_to_ecef)
        self.lanegraph_df = load_pickled_df(
            cfg["preprocessed_data_loader"]["lanegraph"])

    @staticmethod
    def from_config(data_manager: DataManager, cfg: dict) -> "ExtendedMapAPI":
        """Build a ExtendedMapAPI object starting from a config file and a data manager

        :param data_manager: a data manager object ot resolve paths
        :param cfg: the config dict
        :return: a MapAPI object
        """
        raster_cfg = cfg["raster_params"]
        dataset_meta_key = raster_cfg["dataset_meta_key"]

        semantic_map_filepath = data_manager.require(
            raster_cfg["semantic_map_key"])
        dataset_meta = load_metadata(data_manager.require(dataset_meta_key))
        world_to_ecef = np.array(
            dataset_meta["world_to_ecef"], dtype=np.float64)

        return ExtendedMapAPI(semantic_map_filepath, world_to_ecef, cfg)

    @staticmethod
    @no_type_check
    def is_traffic_control_element(element: MapElement) -> bool:
        """
        Check whether an element is a valid traffic control element

        Args:
            element (MapElement): a proto element

        Returns:
            bool: True if the element is a valid traffic control element
        """
        return bool(element.element.HasField("traffic_control_element"))

    @lru_cache(maxsize=CACHE_SIZE)
    def get_traffic_control_element_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a traffic control element given its id
        lru_cached for O(1) access

        Args:
            element_id (str): traffic control element id

        Returns:
            dict: a dict with the polygon coordinates as an (Nx3) XYZ array
        """
        element = self[element_id]
        assert self.is_traffic_control_element(element)
        traffic_element = element.element.traffic_control_element

        xyz = self.unpack_deltas_cm(
            traffic_element.points_x_deltas_cm,
            traffic_element.points_y_deltas_cm,
            traffic_element.points_z_deltas_cm,
            traffic_element.geo_frame,
        )

        return {"xyz": xyz}

    def traffic_control_element_is_type(self, element_id: str, type: str) -> bool:
        """
        Check if the traffic control element is of given type

        Available types:
            * stop_sign
            * traffic_light
            * parking_zone

        Args:
            element_id (str): the id (utf-8 encode) of the element

        Returns:
            True if the element is a stop sign
        """
        element = self[element_id]
        if not element.element.HasField("traffic_control_element"):
            raise Exception(
                f"element: {element_id} is not a traffic control element!")

        traffic_el = element.element.traffic_control_element
        return traffic_el.HasField(type) is True

    @staticmethod
    @no_type_check
    def is_traffic_light(element: MapElement) -> bool:
        """
        Check whether an element is a valid traffic control element

        Args:
            element (MapElement): a proto element

        Returns:
            bool: True if the element is a valid traffic light
        """
        return bool(element.element.HasField("traffic_control_element")) and bool(element.element.traffic_control_element.HasField("traffic_light"))

    @lru_cache(maxsize=CACHE_SIZE)
    def get_traffic_light_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a traffic light given its id
        lru_cached for O(1) access

        Args:
            element_id (str): traffic light element id

        Returns:
            dict: a dict with the polygon coordinates as an (Nx3) XYZ array
        """
        element = self[element_id]
        assert self.is_traffic_light(element)
        traffic_element = element.element.traffic_control_element

        xyz = self.unpack_deltas_cm(
            traffic_element.points_x_deltas_cm,
            traffic_element.points_y_deltas_cm,
            traffic_element.points_z_deltas_cm,
            traffic_element.geo_frame,
        )

        return {"xyz": xyz}

    @lru_cache(maxsize=CACHE_SIZE)
    def get_element_by_id(self, element_id: str) -> "MapElement":
        """Get an element by its element id

        Args:
            element_id (str): element id
        """
        return self[element_id]

    def get_tl_bounds(self) -> dict:
        """
        For each tl element returns bounds [[min_x, min_y],[max_x, max_y]] and proto ids
        Coords are computed by the MapAPI and, as such, are in the world ref system.

        Returns:
            dict: keys are classes of elements, values are dict with `bounds` and `ids` keys
        """
        tl_ids = []
        tl_bounds = np.empty((0, 2, 2), dtype=np.float)

        for element in self.elements:
            element_id = MapAPI.id_as_str(element.id)

            if self.is_traffic_light(element):
                traffic_light = self.get_traffic_light_coords(element_id)
                x_min, y_min = np.min(traffic_light["xyz"], axis=0)[:2]
                x_max, y_max = np.max(traffic_light["xyz"], axis=0)[:2]

                tl_bounds = np.append(
                    tl_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0,
                )
                tl_ids.append(element_id)

        return {
            "bounds": tl_bounds,
            "ids": tl_ids
        }

    @no_type_check
    def get_tl_by_indices(self, tl_indices: np.ndarray, tl_ids: list) -> List["MapElement"]:
        tls = []
        for idx in tl_indices:
            current_tl_id = tl_ids[idx]
            current_tl = self.get_element_by_id(current_tl_id)
            tls.append(current_tl)
        return tls

    @no_type_check
    def get_tl_face_ids(self, element: MapElement) -> Set["str"]:
        tl_element = element.element
        return set([self.id_as_str(face_state) for face_state in tl_element.traffic_control_element.traffic_light.face_states])

    @no_type_check
    def get_tl_id_of_face_id(self, face_id: str, traffic_lights: list) -> str:
        traffc_light_id = None
        for element in traffic_lights:
            face_states = element.element.traffic_control_element.traffic_light.face_states
            face_ids = [self.id_as_str(e) for e in face_states]
            if face_id in face_ids:
                traffc_light_id = self.id_as_str(element.id)
        return traffc_light_id

    @lru_cache(maxsize=CACHE_SIZE)
    def get_traffic_face_color(self, element_id: str) -> Tuple[TLFacesColors, None]:
        """
        Gets the color of traffic face if available

        Args:
            element_id (str): the id (utf-8 encode) of the element
        Returns:
            True if the element is a traffic light face, False otherwise
        """

        for color in TLFacesColors:
            color_name = color.name
            if self.is_traffic_face_color(element_id, color_name.lower()):
                return color
        return None

    def get_nodes_for_lane(self, element_id: str) -> List["MapElement"]:
        """Fetch nodes fo a specific lane

        Args:
            element_id (str): element id of lane

        Returns:
            MapElement: returns corresponding lane nodes
        """
        lane = self.get_element_by_id(element_id)
        assert self.is_lane(lane), "Element is not a lane!"

        # get parent_segment of lane
        parent_segment_or_junction_id = lane.element.lane.id
        parent_segment_or_junction = self.get_element_by_id(
            parent_segment_or_junction_id)

        start_node_id = ""
        end_node_id = ""

        # case 1: segment
        if parent_segment_or_junction.element.HasField("segment"):
            start_node_id = parent_segment_or_junction.element.segment.start_node.id
            end_node_id = parent_segment_or_junction.element.segment.end_node_id.id
        elif parent_segment_or_junction.element.HasField("junction"):
            road_network_nodes = parent_segment_or_junction.element.junction.road_network_nodes
            road_network_node_ids = [node.id for node in road_network_nodes]
        else:
            raise ValueError(
                f"Element {parent_segment_or_junction.id.id} is not from type junction or segment ")

    def get_node_coordinates(self, element_id: str) -> np.ndarray:
        """Get xyz Coordinates of a node transformed to the world_coordinate system

        Args:
            element_id (str): element id of the node

        Returns:
            np.ndarray: xyz coords
        """
        element = self[element_id]
        assert self.is_node(element)

        xyz = self.transform_location(element.element.node.location)
        return xyz

    def get_nodes_coordinates(self) -> dict:
        """Returns a Dictonary of ids with a list of corresponding node coordinates

        Returns:
            np.ndarray: _description_
        """

        nodes = []
        for element in self.elements:
            if self.is_node(element):
                nodes.append(element)

        coords_list = []
        id_list = []
        for node in nodes:
            id = self.id_as_str(node.id)
            coords = self.get_node_coordinates(id)
            coords_list.append(coords)
            id_list.append(id)

        return {"ids": id_list, "coords": coords_list}

    def get_junctions_coords(self, with_tl_only=True) -> dict:
        junctions = self.get_elements_from_layer("junction")
        branches_df = pd.read_csv("dataset/preprocessed_info/junction_branch_to_lane.csv",  encoding="ascii")
        if with_tl_only:
            junctions = self.filter_tl_junctions(junctions)

        coords_list = []
        id_list = []
        for junction in junctions:
            id = self.id_as_str(junction.id)
            if id in branches_df["junction_id"].unique():
                coords = self.get_junction_coords(id)
                coords_list.append(coords)
                id_list.append(id)

        return {"ids": id_list, "coords": coords_list}

    @lru_cache(maxsize=CACHE_SIZE)
    def get_junction_coords(self, element_id: str) -> np.ndarray:
        """
        Get XYZ coordinates of the bounding box of a junction

        Args:
            element_id (str): junction element id

        Returns:
            np.ndarray: an array with center coordinates as XYZ array
        """
        element = self[element_id]
        assert self.is_junction(element)

        xyz = self.transform_location(element.bounding_box.south_west)
        # north_east = self.transform_location(element.bounding_box.north_east)
        # xyz = np.array([south_west, north_east])
        return xyz

    @no_type_check
    def transform_location(self, location: GeoLocation) -> np.ndarray:
        frame_lat, frame_lng = self._undo_e7(
            location.lat_e7), self._undo_e7(location.lng_e7)
        xyz = np.array(pm.enu2ecef(0, 0, location.altitude_cm /
                       100, frame_lat, frame_lng, 0))
        xyz = transform_point(xyz, self.ecef_to_world)
        return xyz

    @staticmethod
    @no_type_check
    def is_junction(elem):
        return elem.element.HasField("junction")

    @lru_cache(maxsize=CACHE_SIZE)
    def get_junctions(self, elements: np.ndarray):
        return [elem for elem in elements if ExtendedMapAPI.is_junction(elem)]

    @staticmethod
    @no_type_check
    def is_node(elem):
        return elem.element.HasField("node")

    @lru_cache(maxsize=CACHE_SIZE)
    def get_nodes(self, elements: np.ndarray):
        return [elem for elem in elements if ExtendedMapAPI.is_node(elem)]

    def get_lane_successors(self, element_id: str) -> List[str]:
        """get the successor lane of the current lane

        Args:
            element_id (str): element id of the current lane

        Returns:
            list: element id of the successor lane, otherwise None is returned
        """
        lane_element = self.get_element_by_id(element_id)
        assert self.is_lane(lane_element)
        return self.lanegraph_df[self.lanegraph_df["lane_id"] == element_id]["successor"].iloc[0]

    def get_lane_predecessors(self, element_id: str) -> List[str]:
        """get the predecessor lane of the current lane

        Args:
            element_id (str): element id of the current lane

        Returns:
            list: element id of the predecessor lane, otherwise None is returned
        """
        lane_element = self.get_element_by_id(element_id)
        assert self.is_lane(lane_element)
        return self.lanegraph_df[self.lanegraph_df["lane_id"] == element_id]["predecessor"].iloc[0]

    def element_of_type(self, elem: MapElement, layer_name: str) -> bool:
        """Check if element is of specific type (layer_name)

        Args:
            elem (MapElement): given element to check type 
            layer_name (str): type to check for

        Returns:
            bool: true if element of type layer_name
        """
        return elem.element.HasField(layer_name)

    @no_type_check
    def get_elements_from_layer(self, layer_name: str) -> List["MapElement"]:
        """Returns element of specific layer type. 
        Available fields are:
            * junction
            * lane
            * node
            * traffic_control_element
            * segment

        Args:
            layer_name (str): layer name to get elements of

        Returns:
            list: list of elements of specific type
        """
        return [elem for elem in self.elements if self.element_of_type(elem, layer_name)]

    @no_type_check
    def get_element_ids_from_layer(self, layer_name: str) -> List[str]:
        """Returns element ids of specific layer type.
        Available fields are:
            * junction
            * lane
            * node
            * traffic_control_element
            * segment

        Args:
            layer_name (str): layer name to get elements of

        Returns:
            list: list of element ids of specific type
        """
        element_list = self.get_elements_from_layer(layer_name)
        return [self.id_as_str(element.id) for element in element_list]

    @no_type_check
    def filter_tl_junctions(self, junctions: List["MapElement"]) -> List["MapElement"]:
        """Filters for junctions that are controlled by traffic lights

        Returns:
            List["MapElement"]: junctions with traffic lights
        """
        tl_junctions = []

        for junction in junctions:
            traffic_control_element_ids = []
            is_tl_junction = False

            # get traffic control elements if available
            for element in junction.element.junction.traffic_control_elements:
                traffic_control_element_ids.append(element.id)

            # check for each element if traffic controll element is a tl
            for id in traffic_control_element_ids:
                traffic_control_element = self.get_element_by_id(id)
                if self.is_traffic_light(traffic_control_element):
                    is_tl_junction = True

            if is_tl_junction:
                tl_junctions.append(junction)

        return tl_junctions

    def map_tl_faces_to_color(self, tl_face_ids: np.ndarray):
        """Map active traffic light face ids to a traffic light color 
        that represent the state of a traffic light

        Args:
            tf_faces (np.ndarray): array of traffic light faces
        Returns:
            (TLFacesColors.name): name of face color
        """
        assert len(
            tl_face_ids) <= 2, f"Number of active tl_face_ids for a traffic light must be under three"

        colors = []
        for face_id in tl_face_ids:
            colors.append(self.get_traffic_face_color(face_id).name)

        if len(colors) == 1:
            return colors[0]
        if len(colors) == 2:  # state mus be red and yellow
            return colors[0]  # return TLFacesColors.RED anyway

    def find_recursiv_predecessors(self, lane_id: str, range: int, center: np.ndarray, precedent_lane_ids: set = set()) -> set:
        """Recursive finds get_precedent_lane_segments("7/jP", mapAPI, 50) lane segments in a specific area (limited by range)

        Implementation based on: [Depth-First Search](https://www.koderdojo.com/blog/depth-first-search-in-python-recursive-and-non-recursive-programming)

        Args:
            lane_id (str): lane id
            range (int): range limiter (radius)
            center (np.ndarray): center of circle area
            precedent_lane_ids (set, optional): precedent lane ids. Defaults to set().

        Returns:
            set: returns all precedent lane ids in a specific area
        """
        precedent_lane_ids = precedent_lane_ids.union(set([lane_id]))
        lane_bounds = self.bounds_info["lanes"]

        for predecessor in self.get_lane_predecessors(lane_id):
            predecessor_lane_idx = lane_bounds["ids"].index(predecessor)
            predecessor_lane_bounds = np.array(
                [lane_bounds["bounds"][predecessor_lane_idx]])
            if predecessor not in precedent_lane_ids and len(indices_in_bounds(center, predecessor_lane_bounds, range)) != 0:
                precedent_lane_ids = self.find_recursiv_predecessors(
                    predecessor, range, center, precedent_lane_ids)

        return precedent_lane_ids

    def get_precedent_lane_segments(self, lane_id: str, range: int) -> set:
        """Returns list of preceding lane segment ids of given lane id in a specific range

        Args:
            lane_id (str): given lane id
            range (int): range to get predecessor lane ids for

        Returns:
            list: preceding lane segment ids
        """
        lane_coords = self.get_lane_coords(lane_id)
        portal = np.array(
            [lane_coords["xyz_left"][0, 0:2], lane_coords["xyz_right"][0, 0:2]])

        portal_center = calculate_center(np.array(
            [portal[0, 0], portal[1, 0]]), np.array([portal[0, 1], portal[1, 1]]))

        return self.find_recursiv_predecessors(lane_id, range, portal_center)

    def get_lane_stopping_point(self, lane_id: str) -> np.ndarray:
        """get xy-coordinate of lane stopping point

        Args:
            lane_id (str): id of lane

        Returns:
            np.ndarray: xy-coordinate of stopping point
        """
        lane_coords = self.get_lane_coords(lane_id)
        portal = np.array(
            [lane_coords["xyz_left"][-1, 0:2], lane_coords["xyz_right"][-1, 0:2]])

        return calculate_center(np.array(
            [portal[0, 0], portal[1, 0]]), np.array([portal[0, 1], portal[1, 1]]))

    def get_tl_coordinate(self, tl_id: str) -> np.ndarray:
        """get bottom-left xy-coordinate of tl

        Args:
            tl_id (str): id of traffic light

        Returns:
            np.ndarray: xy-coordinate of tl
        """
        tl_bounds = self.get_tl_bounds()
        idx = tl_bounds["ids"].index(tl_id)
        return tl_bounds["bounds"][idx][0,:] # get only bottom left coordinate

        

        return 

    def exists_in_semantic_map(self, element_id: str) -> bool:
        """Check if element id exists in semantic map

        Args:
            element_id (str): element id

        Returns:
            bool: element exists?
        """
        return element_id in self.ids_to_el.keys()

    def tl_of_junction(self, tl_id: str, junction_id: str) -> bool:
        """Checks if traffic light is part of junction

        Args:
            tl_id (str): id of traffic light
            junction_id (str): junction id

        Returns:
            bool: tl exist in junction
        """
        junction_element = self.get_element_by_id(junction_id)
        junction_traffic_control_element_ids = junction_element.element.junction.traffic_control_elements
        junction_traffic_control_element_ids = [self.id_as_str(
            id) for id in junction_traffic_control_element_ids]
        return tl_id in junction_traffic_control_element_ids

    def get_lane_tl_relation(self) -> Dict["str", set]:
        """Get the relation traffic lights to a lane for
        a given semantic map.
        Returns:
            Dict["str", set]: lane_id -> set of traffic light ids
        """

        tl_ids: list = self.get_tl_bounds()["ids"]
        lane_ids: list = self.bounds_info["lanes"]["ids"]

        tls = [self.get_element_by_id(id) for id in tl_ids]
        lane_tls_dict = dict()

        for lane_id in lane_ids:
            lane_traffic_controll_ids = self.get_lane_traffic_control_ids(
                lane_id)

            # list of traffic controll element of type traffic light next to a lane
            lane_traffic_lights = []

            for element_id in lane_traffic_controll_ids:
                control_element_id = self.get_tl_id_of_face_id(
                    element_id, tls)
                if control_element_id is not None:  # traffic light for a given face id found
                    lane_traffic_lights.append(control_element_id)

            if len(lane_traffic_lights) != 0:
                lane_tls_dict[lane_id] = set(lane_traffic_lights)

        return lane_tls_dict


    def has_adjacent_lane_change_right(self, lane_id: str)-> bool:
        """Checks if the lane has adjacent lane on right change.

        Used criteria:
            * righternmoast lane
            * no further adjacent_lane_change_right available

        Returns:
            bool: if adjacent lane on right change exist
        """
        lane = self.get_element_by_id(lane_id)
        assert self.is_lane(lane)

        return not (lane.element.lane.adjacent_lane_change_right.id == b'')
    
    def has_adjacent_lane_change_left(self, lane_id: str)-> bool:
        """Checks if the lane has adjacent lane on right change.

        Used criteria:
            * righternmoast lane
            * no further adjacent_lane_change_right available

        Returns:
            bool: if adjacent lane on right change exist
        """
        lane = self.get_element_by_id(lane_id)
        assert self.is_lane(lane)

        return not (lane.element.lane.adjacent_lane_change_left.id == b'')
    

    def get_turn_type(self, lane_id: str) -> int:
        """Returns the turn type of a sequence head lane id

        Args:
            lane_id (str): id of lane

        Returns:
            int: turn type
        """
        if not self.has_adjacent_lane_change_left(lane_id):
            return TURN.LEFT.value
        if not self.has_adjacent_lane_change_right(lane_id):
            return TURN.RIGHT.value
        return TURN.NORMAL.value
