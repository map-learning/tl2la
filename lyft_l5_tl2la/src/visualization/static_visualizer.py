# ------------------------------------------------------------------------
# Visualizer based on Bokeh to render Map features and semantics of 
# Lyft's HD map
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from L5Kit (https://github.com/woven-planet/l5kit)
# Copyright (c) 2022 Woven by Toyota. All Rights Reserved.
#
# Based on files:
#   - l5kit.visualization.visualizer.py
#   - l5kit.visualization.zarr_utils.py
# ------------------------------------------------------------------------


import pickle
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set

import bokeh.io
import bokeh.plotting
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import LayoutDOM
from bokeh.models import BasicTicker, ColorBar, HoverTool, LinearColorMapper
from bokeh.palettes import Plasma
from bokeh.plotting import ColumnDataSource
from l5kit.data import DataManager
from l5kit.visualization.visualizer.common import CWVisualization
from src.common.helper import load_pickled_df, reverse_dict_of_sets
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.visualization.tl_visualization import (JunctionVisualization,
                                                LaneTLRelationVisualization,
                                                LaneVisualization,
                                                NodeVisualization,
                                                StaticVisualization,
                                                TLVisualization)


class StaticVisualizer:
    def __init__(self, cfg: dict, dm: DataManager) -> None:
        self.cfg = cfg
        self.dm = dm

        self.frame_lane_df: pd.DataFrame = None

    def _calculate_center(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Calculate center of two given points

        Args:
            xs (np.ndarray): x-pos of points
            ys (np.ndarray): y-pos of points

        Returns:
            np.ndarray: center point
        """
        c1 = xs[0] + 0.5 * (xs[1]-xs[0])
        c2 = ys[0] + 0.5 * (ys[1]-ys[0])

        return np.array([c1, c2])

    def load_df(self, path: str) -> pd.DataFrame:
        """Returns dataframe of pickled file

        Args:
            path (str): path to pickle file

        Returns:
            pd.DataFrame: unpacked pickeld dataframe
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_semantic_visualisation(self, mapAPI: ExtendedMapAPI, **kwargs) -> StaticVisualization:
        """Get visualisation objects for static map.

        Keyword Args:
        * heatmap (bool): Visualization of the frequency of lane usage by the ego vehicle
        """
        #################
        # plot lane
        lane_ids: list = mapAPI.bounds_info["lanes"]["ids"]
        lane_id_to_scene_ids_df = None

        if "heatmap" in kwargs.keys() and kwargs["heatmap"]:
            self.frame_lane_df = self.load_df(
                # path=self.cfg["preprocessed_data_loader"]["ego_frame_lane_mapping"])
                path="dataset/preprocessed_info/frame_lane_df_trainfull.pkl")
            self.branches_df = pd.read_csv(
                self.cfg["preprocessed_data_loader"]["junction_branch_to_lane"], encoding="ascii")

            merged_df = self.frame_lane_df.join(
                self.branches_df.set_index('lane_id'), on=['lane_id'])
            merged_df = merged_df.fillna("unknown")

            scene_ids_per_lane = merged_df.groupby(
                "lane_id")["scene_idx"].unique()
            lane_id_to_scene_ids_df = pd.DataFrame(
                columns=["lane_id", "scene_indices"])
            lane_id_to_scene_ids_df["scene_indices"] = lane_id_to_scene_ids_df["scene_indices"].astype(
                'object')
            lane_id_to_scene_ids_df["scene_indices"] = scene_ids_per_lane.values.tolist(
            )
            lane_id_to_scene_ids_df["lane_id"] = merged_df["lane_id"].unique(
            ).tolist()
            lane_id_to_scene_ids_df["trips"] = lane_id_to_scene_ids_df["scene_indices"].apply(
                lambda x: len(x))

            jet = plt.get_cmap('plasma')
            cNorm = colors.Normalize(
                vmin=0, vmax=lane_id_to_scene_ids_df["trips"].max())
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        lanes_vis: List[LaneVisualization] = []

        for lane_id in lane_ids:
            lane_colour = "gray"
            if lane_id_to_scene_ids_df is not None:
                current_row = lane_id_to_scene_ids_df[lane_id_to_scene_ids_df["lane_id"] == lane_id]
                if len(current_row) > 0:
                    lane_colour = colors.to_hex(list(scalarMap.to_rgba(
                        current_row["trips"].iloc[0])))

            lane_coords = mapAPI.get_lane_coords(lane_id)
            left_lane = lane_coords["xyz_left"][:, :2]
            right_lane = lane_coords["xyz_right"][::-1, :2]

            lanes_vis.append(LaneVisualization(xs=np.hstack((left_lane[:, 0], right_lane[:, 0])),
                                               ys=np.hstack(
                                                   (left_lane[:, 1], right_lane[:, 1])),
                                               color=lane_colour,
                                               id=lane_id))

        #################
        # plot traffic light to lane relation: 
        lane_tl_vis: List[LaneTLRelationVisualization] = []

        # get corresponding traffic lights of lanes
        lane_tls_dict = mapAPI.get_lane_tl_relation()
        tl_lanes_dict = reverse_dict_of_sets(lane_tls_dict)

        for lane_id in lane_tls_dict:
            corresponding_tl_ids = lane_tls_dict[lane_id]

            predecessor_lanes = mapAPI.get_lane_predecessors(lane_id)
            # has_predecessor = any(predecessor in lane_tls_dict.keys()
            #                       for predecessor in predecessor_lanes)
            # if has_predecessor:
            #     continue

            for tl_id in corresponding_tl_ids:
                has_predecessor = any(predecessor in tl_lanes_dict[tl_id]
                                      for predecessor in predecessor_lanes)
                if has_predecessor:
                    continue

                # calculate points of lane portals (start and end)

                # start points
                tl_coords = mapAPI.get_traffic_light_coords(tl_id)
                p_start = self._calculate_center(
                    tl_coords["xyz"][:, 0], tl_coords["xyz"][:, 1])

                # end point
                lane_coords = mapAPI.get_lane_coords(lane_id)
                portal_1 = np.array(
                    [lane_coords["xyz_left"][-1, 0:2], lane_coords["xyz_right"][-1, 0:2]])

                portal1_center = self._calculate_center(np.array(
                    [portal_1[0, 0], portal_1[1, 0]]), np.array([portal_1[0, 1], portal_1[1, 1]]))
                p_end = portal1_center

                lane_tl_vis.append(
                    LaneTLRelationVisualization(xs=np.array([p_start[0], p_end[0]]),
                                                ys=np.array([p_start[1], p_end[1]]))
                )
                
        #################
        # plot predicted traffic light to lane relation: 
        predicted_lane_tl_vis: List[LaneTLRelationVisualization] = []
        gt_lane_tl_vis: List[LaneTLRelationVisualization] = []

        if "predicted_assocs" in kwargs.keys() and kwargs["predicted_assocs"]:
            # get corresponding traffic lights of lanes
            path="dataset/results/conservative/groundtruth_vs_prediction_notna.pkl"
            groundtruth_vs_prediction_notna = load_pickled_df(path)
            
            predicted_lane_tls_dict = {lane_id: set() for lane_id in groundtruth_vs_prediction_notna["lane_id"].unique()}
            gt_lane_tls_dict = {lane_id: set() for lane_id in groundtruth_vs_prediction_notna["lane_id"].unique()}
            for idx, row in groundtruth_vs_prediction_notna.iterrows():
                if row["predicted_relation"] > 0:
                    predicted_lane_tls_dict[row["lane_id"]].add(row["tl_id"])
                if row["relation"] > 0:
                    gt_lane_tls_dict[row["lane_id"]].add(row["tl_id"])
 
            for lane_id in predicted_lane_tls_dict:
                corresponding_tl_ids = predicted_lane_tls_dict[lane_id]

                predecessor_lanes = mapAPI.get_lane_predecessors(lane_id)

                for tl_id in corresponding_tl_ids:
                    has_predecessor = any(predecessor in tl_lanes_dict[tl_id]
                                        for predecessor in predecessor_lanes)
                    if has_predecessor:
                        continue

                    # calculate points of lane portals (start and end)

                    # start points
                    tl_coords = mapAPI.get_traffic_light_coords(tl_id)
                    p_start = self._calculate_center(
                        tl_coords["xyz"][:, 0], tl_coords["xyz"][:, 1])

                    # end point
                    lane_coords = mapAPI.get_lane_coords(lane_id)
                    portal_1 = np.array(
                        [lane_coords["xyz_left"][-1, 0:2], lane_coords["xyz_right"][-1, 0:2]])

                    portal1_center = self._calculate_center(np.array(
                        [portal_1[0, 0], portal_1[1, 0]]), np.array([portal_1[0, 1], portal_1[1, 1]]))
                    p_end = portal1_center
                    
                    offset = -0.1
                    predicted_lane_tl_vis.append(
                        LaneTLRelationVisualization(xs=np.array([p_start[0]+offset, p_end[0]+offset]),
                                                    ys=np.array([p_start[1]+offset, p_end[1]+offset]))
                    )
            for lane_id in gt_lane_tls_dict:
                corresponding_tl_ids = gt_lane_tls_dict[lane_id]

                predecessor_lanes = mapAPI.get_lane_predecessors(lane_id)

                for tl_id in corresponding_tl_ids:
                    has_predecessor = any(predecessor in tl_lanes_dict[tl_id]
                                        for predecessor in predecessor_lanes)
                    if has_predecessor:
                        continue

                    # calculate points of lane portals (start and end)

                    # start points
                    tl_coords = mapAPI.get_traffic_light_coords(tl_id)
                    p_start = self._calculate_center(
                        tl_coords["xyz"][:, 0], tl_coords["xyz"][:, 1])

                    # end point
                    lane_coords = mapAPI.get_lane_coords(lane_id)
                    portal_1 = np.array(
                        [lane_coords["xyz_left"][-1, 0:2], lane_coords["xyz_right"][-1, 0:2]])

                    portal1_center = self._calculate_center(np.array(
                        [portal_1[0, 0], portal_1[1, 0]]), np.array([portal_1[0, 1], portal_1[1, 1]]))
                    p_end = portal1_center
                    
                    offset = 0.1
                    gt_lane_tl_vis.append(
                        LaneTLRelationVisualization(xs=np.array([p_start[0]+offset, p_end[0]+offset]),
                                                    ys=np.array([p_start[1]+offset, p_end[1]+offset]))
                    )

        #################
        # plot traffic lights: 

         # reverse lane_tls_dict
        tl_lanes_dict = dict()
        for lane_id, tl_set in lane_tls_dict.items():
            for tl_id in tl_set:
                if tl_id not in tl_lanes_dict.keys():
                    tl_lanes_dict[tl_id] = {lane_id}
                else:
                    tl_lanes_dict[tl_id] = tl_lanes_dict[tl_id].union({
                                                                      lane_id})

        tl_ids = mapAPI.get_tl_bounds()["ids"]

        tl_vis: List[TLVisualization] = []

        for tl_id in tl_ids:
            tl_color = "grey"
            tl_color_state = "unknown"

            tl_coords = mapAPI.get_traffic_light_coords(tl_id)
            lane_ids = list(tl_lanes_dict.get(tl_id, []))
            tl_element = mapAPI.get_element_by_id(tl_id)
            tl_face_ids = mapAPI.get_tl_face_ids(tl_element)

            center = self._calculate_center(
                tl_coords["xyz"][:, 0], tl_coords["xyz"][:, 1])
            tl_vis.append(TLVisualization(xs=tl_coords["xyz"][:, 0],
                                          ys=tl_coords["xyz"][:, 1],
                                          center_x=center[0],
                                          center_y=center[1],
                                          state_color=tl_color,
                                          state=tl_color_state,
                                          traffic_light_id=tl_id,
                                          face_ids=np.array(
                                              list(tl_face_ids)),
                                          corresponding_lanes=lane_ids))  # lane_ids

        #################
        # plot traffic nodes: 
        node_vis: List[NodeVisualization] = []
        nodes_coords = np.array(mapAPI.get_nodes_coordinates()["coords"])
        nodes_ids = np.array(mapAPI.get_nodes_coordinates()["ids"])

        for idx, node_coords in enumerate(nodes_coords):
            node_vis.append(
                NodeVisualization(
                    x=node_coords[0], y=node_coords[1], id=nodes_ids[idx])
            )

        #################
        # plot junction points: 
        junction_vis: List[JunctionVisualization] = []
        junctions_coords = np.array(mapAPI.get_junctions_coords()["coords"])
        junctions_ids = mapAPI.get_junctions_coords()["ids"]

        for idx, junction_coords in enumerate(junctions_coords):
            junction_vis.append(
                JunctionVisualization(
                    x=junction_coords[0], y=junction_coords[1], id=junctions_ids[idx])
            )

        #################
        # plot crosswalks
        crosswalks_coords = mapAPI.bounds_info["crosswalks"]["bounds"]
        crosswalks_vis: List[CWVisualization] = []

        for coords in crosswalks_coords:
            crosswalks_vis.append(CWVisualization(xs=coords[:, 0],
                                                  ys=coords[:, 1],
                                                  color="yellow"))

        return StaticVisualization(lanes=lanes_vis,
                                   crosswalks=crosswalks_vis,
                                   trafficLights=tl_vis,
                                   lane_tl_relations=lane_tl_vis,
                                   predicted_lane_tl_relations=predicted_lane_tl_vis,
                                   gt_lane_tl_relations=gt_lane_tl_vis,
                                   nodes=node_vis,
                                   tl_junctions=junction_vis
                                   )

    def _visualization_list_to_dict(self, visualisation_list: List[Any], null_el: Any) -> Dict[str, Any]:
        """Convert a list of NamedTuple into a dict, where:
            - the NamedTuple fields are the dict keys;
            - the dict value are lists;

            :param visualisation_list: a list of NamedTuple
            :param null_el: an element to be used as null if the list is empty (it can crash visualisation)
            :return: a dict with the same information
        """
        visualisation_list = visualisation_list if len(
            visualisation_list) else [null_el]
        visualisation_dict: DefaultDict[str, Any] = defaultdict(list)

        keys_set: Set[str] = set(visualisation_list[0]._asdict().keys())
        for el in visualisation_list:
            for k, v in el._asdict().items():
                if k not in keys_set:
                    raise ValueError(
                        "keys set is not consistent between elements in the list")
                visualisation_dict[k].append(v)
        return dict(visualisation_dict)

    def visualize(self, static_vis: StaticVisualization, **kwargs) -> LayoutDOM:
        """Visualise static environment using Bokeh.
        """

        tl_hover = HoverTool(
            mode="mouse",
            names=["trafficLights"],
            tooltips=[
                ("xPos", "@center_x{0.00}"),
                ("yPos", "@center_y{0.00}"),
                ("State", "@state"),
                ("TL Id", "@traffic_light_id"),
                ("Face Ids", "@face_ids"),
                ("Lanes", "@corresponding_lanes")
            ],
        )

        lane_hover = HoverTool(
            mode="mouse",
            names=["lanes"],
            tooltips=[
                ("(x,y)", "($x, $y)"),
                ("Lane Id", "@id")
            ],
        )

        junction_hover = HoverTool(
            mode="mouse",
            names=["tl_junctions"],
            tooltips=[
                ("xPos", "@x{0.00}"),
                ("yPos", "@y{0.00}"),
                ("Id", "@id"),
            ],
        )

        node_hover = HoverTool(
            mode="mouse",
            names=["nodes"],
            tooltips=[
                ("xPos", "@x{0.00}"),
                ("yPos", "@y{0.00}"),
                ("Id", "@id"),
            ],
        )

        out: Dict[str, ColumnDataSource] = None

        # we need to ensure we have something otherwise js crashes

        lanes_dict = self._visualization_list_to_dict(static_vis.lanes, LaneVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                          color="black", id="none"))

        crosswalk_dict = self._visualization_list_to_dict(static_vis.crosswalks, CWVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                                 color="black"))
        tl_dict = self._visualization_list_to_dict(static_vis.trafficLights, TLVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                             center_x=0, center_y=0,
                                                                                             state_color="grey", state="unknown",
                                                                                             traffic_light_id="none", face_ids=np.empty(0),
                                                                                             corresponding_lanes=["unknown"]))

        lane_tl_dict = self._visualization_list_to_dict(
            static_vis.lane_tl_relations, LaneTLRelationVisualization(xs=np.zeros((2,), dtype=float), ys=np.zeros((2,), dtype=float)))
        
        predicted_lane_tl_dict = self._visualization_list_to_dict(
            static_vis.predicted_lane_tl_relations, LaneTLRelationVisualization(xs=np.zeros((2,), dtype=float), ys=np.zeros((2,), dtype=float)))
        
        gt_lane_tl_dict = self._visualization_list_to_dict(
            static_vis.gt_lane_tl_relations, LaneTLRelationVisualization(xs=np.zeros((2,), dtype=float), ys=np.zeros((2,), dtype=float)))

        node_dict = self._visualization_list_to_dict(
            static_vis.nodes, NodeVisualization(x=0., y=0., id="none"))

        junction_dict = self._visualization_list_to_dict(
            static_vis.tl_junctions, JunctionVisualization(x=0., y=0., id="unknown"))

        out = dict(lanes=ColumnDataSource(lanes_dict),
                   crosswalks=ColumnDataSource(crosswalk_dict),
                   traffic_lights=ColumnDataSource(tl_dict),
                   lane_tl_relations=ColumnDataSource(lane_tl_dict),
                   predicted_lane_tl_relations=ColumnDataSource(predicted_lane_tl_dict),
                   gt_lane_tl_relations=ColumnDataSource(gt_lane_tl_dict),
                   nodes=ColumnDataSource(node_dict),
                   tl_junctions=ColumnDataSource(junction_dict))

        scene_fig = bokeh.plotting.figure(
            title="Static Environment of Dataset",
            match_aspect=True,
            tools=["pan", "wheel_zoom",
                   tl_hover,
                   lane_hover,
                   junction_hover,
                   node_hover,
                   "save", "reset"],
            active_scroll="wheel_zoom",
            align='center',
            plot_width=1000,
            plot_height=1000
        )

        scene_fig.xgrid.grid_line_color = None
        scene_fig.ygrid.grid_line_color = None

        scene_fig.patches(line_width=0, alpha=0.5, name="lanes",
                          color="color", source=out["lanes"])
        scene_fig.patches(line_width=0, alpha=0.5, color="#B5B50D",
                          source=out["crosswalks"])

        # visualize tls state
        scene_fig.ellipse(x="center_x", y="center_y", width=2, height=2, line_width=2, color="state_color",
                          source=out["traffic_lights"], name="trafficLights")

        # visualize relation
        scene_fig.multi_line(xs="xs", ys="ys",
                             line_width=2, color="#8073ac", source=out["lane_tl_relations"], legend_label="lane_tl_relations")
        
        # visualize predicted relation
        if "predicted_assocs" in kwargs.keys() and kwargs["predicted_assocs"]:
            scene_fig.multi_line(xs="xs", ys="ys",
                                line_width=2, color="#212591", source=out["predicted_lane_tl_relations"], legend_label="predicted_lane_tl_relations")
            # visualize gt relation
            scene_fig.multi_line(xs="xs", ys="ys",
                                line_width=2, color="#a80d10", source=out["gt_lane_tl_relations"], legend_label="ground_truth_lane_tl_relations")
        
        # visualize tls geometry
        scene_fig.patches(line_width=2, color="black",
                          source=out["traffic_lights"])

        # visualize nodes
        scene_fig.ellipse(x="x", y="y", width=2, height=2, line_width=2, color="black",
                          source=out["nodes"], name="nodes", legend_label="nodes")

        # visualize junctions
        scene_fig.ellipse(x="x", y="y", width=3, height=3, line_width=2, color="orange",
                          source=out["tl_junctions"], name="tl_junctions", legend_label="tl_junctions")

        scene_fig.legend.location = "top_left"
        scene_fig.legend.click_policy = "hide"

        if "heatmap" in kwargs.keys() and kwargs["heatmap"]:
            color_mapper = LinearColorMapper(
                palette=Plasma[11], low=0, high=8187)
            color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                                 location=(0, 0))
            scene_fig.add_layout(color_bar, 'right')

        return scene_fig
