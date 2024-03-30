# ------------------------------------------------------------------------
# Visualizer based on Bokeh to render Map semantics and dynamic information
# of Lyft's HD map and dataset.
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


from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set

import bokeh.io
import bokeh.plotting
import numpy as np
from bokeh.layouts import LayoutDOM, column
from bokeh.models import CustomJS, HoverTool, Slider
from bokeh.plotting import ColumnDataSource
from l5kit.data import ChunkedDataset, DataManager
from l5kit.data.filter import (filter_agents_by_frames,
                               filter_agents_by_labels,
                               filter_tl_faces_by_frames,
                               filter_tl_faces_by_status)
from l5kit.data.labels import PERCEPTION_LABELS
from l5kit.rasterization.box_rasterizer import (get_box_world_coords,
                                                get_ego_as_agent)
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.sampling.agent_sampling import get_relative_poses
from l5kit.visualization.visualizer.common import (AgentVisualization,
                                                   CWVisualization,
                                                   EgoVisualization,
                                                   TrajectoryVisualization)
from l5kit.visualization.visualizer.zarr_utils import COLORS
from src.common.helper import (calculate_center, indices_in_bound,
                               reverse_dict_of_sets)
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.visualization.tl_visualization import (FrameVisualization,
                                                LaneTLRelationVisualization,
                                                LaneVisualization,
                                                NodeVisualization,
                                                TLVisualization)


class Visualizer:
    def __init__(self, cfg: dict, dm: DataManager, radius: int) -> None:
        self.cfg = cfg
        self.dm = dm
        self.radius = radius

    def _get_frame_data(self, mapAPI: ExtendedMapAPI, frame: np.ndarray, agents_frame: np.ndarray,
                        tls_frame: np.ndarray) -> FrameVisualization:
        """Get visualisation objects for the current frame.

        :param mapAPI: mapAPI object (used for lanes, crosswalks etc..)
        :param frame: the current frame (used for ego)
        :param agents_frame: agents in this frame
        :param tls_frame: the tls of this frame
        :return: A FrameVisualization object. NOTE: trajectory are not included here
        """
        ego_xy = frame["ego_translation"][:2]

        #################
        # plot lanes
        lane_indices = indices_in_bounds(
            ego_xy, mapAPI.bounds_info["lanes"]["bounds"], self.radius)
        active_tl_face_ids = set(filter_tl_faces_by_status(
            tls_frame, "ACTIVE")["face_id"].tolist())
        lanes_vis: List[LaneVisualization] = []

        for lane_idx in lane_indices:
            lane_id = mapAPI.bounds_info["lanes"]["ids"][lane_idx]

            lane_tl_ids = mapAPI.get_lane_traffic_control_ids(lane_id)
            lane_colour = "gray"
            for tl_id in lane_tl_ids.intersection(active_tl_face_ids):
                lane_colour = COLORS[mapAPI.get_color_for_face(tl_id)]

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

        tl_bounds: np.ndarray = mapAPI.get_tl_bounds()["bounds"]
        tl_ids: list = mapAPI.get_tl_bounds()["ids"]

        tl_indices = indices_in_bounds(
            ego_xy, tl_bounds, self.radius)

        # get corresponding traffic lights of lanes
        lane_tls_dict = mapAPI.get_lane_tl_relation()
        tl_lanes_dict = reverse_dict_of_sets(lane_tls_dict)

        for lane_id in lane_tls_dict:
            corresponding_tl_ids = lane_tls_dict[lane_id]

            predeccor_lanes = mapAPI.get_lane_predecessors(lane_id)

            for tl_id in corresponding_tl_ids:
                has_predecessor = any(predecessor in tl_lanes_dict[tl_id]
                                      for predecessor in predeccor_lanes)
                if has_predecessor:
                    continue

                # calculate points of lane portals (start and end)

                # start points
                tl_coords = mapAPI.get_traffic_light_coords(tl_id)
                p_start = calculate_center(
                    tl_coords["xyz"][:, 0], tl_coords["xyz"][:, 1])

                # end point
                lane_coords = mapAPI.get_lane_coords(lane_id)
                portal_1 = np.array(
                    [lane_coords["xyz_left"][-1, 0:2], lane_coords["xyz_right"][-1, 0:2]])

                portal1_center = calculate_center(np.array(
                    [portal_1[0, 0], portal_1[1, 0]]), np.array([portal_1[0, 1], portal_1[1, 1]]))

                p_end = portal1_center

                lane_tl_vis.append(
                    LaneTLRelationVisualization(xs=np.array([p_start[0], p_end[0]]),
                                                ys=np.array([p_start[1], p_end[1]]))
                )

        #################
        # plot traffic lights:

        tl_bounds = mapAPI.get_tl_bounds()["bounds"]
        tl_ids = mapAPI.get_tl_bounds()["ids"]

        tl_indices = indices_in_bounds(
            ego_xy, tl_bounds, self.radius)

        tl_vis: List[TLVisualization] = []

        for idx in tl_indices:
            tl_color = "grey"
            tl_color_state = "unknown"

            current_tl_id = mapAPI.get_tl_bounds()["ids"][idx]
            current_tl = mapAPI[current_tl_id]
            current_tl_face_ids = mapAPI.get_tl_face_ids(current_tl)

            for face_id in current_tl_face_ids.intersection(active_tl_face_ids):
                tl_color = COLORS[mapAPI.get_color_for_face(face_id)]
                tl_color_state = mapAPI.get_color_for_face(face_id)

            tl_coords = mapAPI.get_traffic_light_coords(
                tl_ids[idx])

            lane_ids = list(tl_lanes_dict.get(current_tl_id, []))

            center = calculate_center(
                tl_coords["xyz"][:, 0], tl_coords["xyz"][:, 1])
            tl_vis.append(TLVisualization(xs=tl_coords["xyz"][:, 0],
                                          ys=tl_coords["xyz"][:, 1],
                                          center_x=center[0],
                                          center_y=center[1],
                                          state_color=tl_color,
                                          state=tl_color_state,
                                          traffic_light_id=current_tl_id,
                                          face_ids=np.array(
                                              list(current_tl_face_ids)),
                                          corresponding_lanes=lane_ids))  # lane_ids

        #################
        # plot traffic nodes:
        node_vis: List[NodeVisualization] = []
        nodes_coords = np.array(mapAPI.get_nodes_coordinates()["coords"])
        nodes_ids = np.array(mapAPI.get_nodes_coordinates()["ids"])

        nodes_indices = indices_in_bound(
            ego_xy, nodes_coords, self.radius)

        relevant_coords = nodes_coords[nodes_indices]
        relevant_ids = nodes_ids[nodes_indices]

        for idx, node_coords in enumerate(relevant_coords):
            node_vis.append(
                NodeVisualization(
                    x=node_coords[0], y=node_coords[1], id=relevant_ids[idx])
            )

        #################
        # plot crosswalks
        crosswalk_indices = indices_in_bounds(
            ego_xy, mapAPI.bounds_info["crosswalks"]["bounds"], self.radius)
        crosswalks_vis: List[CWVisualization] = []

        for idx in crosswalk_indices:
            crosswalk = mapAPI.get_crosswalk_coords(
                mapAPI.bounds_info["crosswalks"]["ids"][idx])
            crosswalks_vis.append(CWVisualization(xs=crosswalk["xyz"][:, 0],
                                                  ys=crosswalk["xyz"][:, 1],
                                                  color="yellow"))
        #################
        # plot ego and agents
        agents_frame = np.insert(agents_frame, 0, get_ego_as_agent(frame))
        box_world_coords = get_box_world_coords(agents_frame)

        # ego
        ego_vis = EgoVisualization(xs=box_world_coords[0, :, 0], ys=box_world_coords[0, :, 1],
                                   color="red", center_x=agents_frame["centroid"][0, 0],
                                   center_y=agents_frame["centroid"][0, 1])

        # agents
        agents_frame = agents_frame[1:]
        box_world_coords = box_world_coords[1:]

        agents_vis: List[AgentVisualization] = []
        for agent, box_coord in zip(agents_frame, box_world_coords):
            label_index = np.argmax(agent["label_probabilities"])
            agent_type = PERCEPTION_LABELS[label_index]
            agents_vis.append(AgentVisualization(xs=box_coord[..., 0],
                                                 ys=box_coord[..., 1],
                                                 color="#1F77B4" if agent_type not in COLORS else COLORS[
                                                     agent_type],
                                                 track_id=agent["track_id"],
                                                 agent_type=PERCEPTION_LABELS[label_index],
                                                 prob=agent["label_probabilities"][label_index]))

        return FrameVisualization(ego=ego_vis, agents=agents_vis, lanes=lanes_vis,
                                  crosswalks=crosswalks_vis, trajectories=[], trafficLights=tl_vis, lane_tl_relations=lane_tl_vis, nodes=node_vis)

    def zarr_to_visualizer_scene(self, scene_dataset: ChunkedDataset, mapAPI: ExtendedMapAPI,
                                 with_trajectories: bool = True) -> List[FrameVisualization]:
        """Convert a zarr scene into a list of FrameVisualization which can be used by the visualiser

        :param scene_dataset: a scene dataset. This must contain a single scene
        :param mapAPI: mapAPI object
        :param with_trajectories: if to enable trajectories or not
        :return: a list of FrameVisualization objects
        """
        if len(scene_dataset.scenes) != 1:
            raise ValueError(
                f"we can convert only a single scene, found {len(scene_dataset.scenes)}")

        frames = scene_dataset.frames
        agents_frames = filter_agents_by_frames(frames, scene_dataset.agents)
        tls_frames = filter_tl_faces_by_frames(frames, scene_dataset.tl_faces)

        frames_vis: List[FrameVisualization] = []
        for frame_idx in range(len(frames)):
            frame = frames[frame_idx]
            tls_frame = tls_frames[frame_idx]

            # TODO: hardcoded threshold, it would be great to have a slider filtering on this
            agents_frame = agents_frames[frame_idx]
            agents_frame = filter_agents_by_labels(agents_frame, 0.1)

            frame_vis = self._get_frame_data(
                mapAPI, frame, agents_frame, tls_frame)

            if with_trajectories:
                traj_vis = self._get_frame_trajectories(
                    frames, agents_frames, agents_frame["track_id"], frame_idx)
                frame_vis = FrameVisualization(ego=frame_vis.ego, agents=frame_vis.agents,
                                               lanes=frame_vis.lanes, crosswalks=frame_vis.crosswalks,
                                               trajectories=traj_vis,
                                               trafficLights=frame_vis.trafficLights,
                                               lane_tl_relations=frame_vis.lane_tl_relations,
                                               nodes=frame_vis.nodes)
            frames_vis.append(frame_vis)

        return frames_vis

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

    def visualize(self, scene_index: int, frames: List[FrameVisualization]) -> LayoutDOM:
        """Visualise a scene using Bokeh.

        :param scene_index: the index of the scene, used only as the title
        :param frames: a list of FrameVisualization objects (one per frame of the scene)
        """

        agent_hover = HoverTool(
            name="Hover Agents",
            mode="mouse",
            names=["agents"],
            tooltips=[
                ("Type", "@agent_type"),
                ("Probability", "@prob{0.00}%"),
                ("Track id", "@track_id"),
            ],
        )

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

        out: List[Dict[str, ColumnDataSource]] = []

        trajectories_labels = np.unique(
            [traj.legend_label for frame in frames for traj in frame.trajectories])

        for frame_idx, frame in enumerate(frames):
            # we need to ensure we have something otherwise js crashes
            ego_dict = self._visualization_list_to_dict([frame.ego], EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                      color="black", center_x=0,
                                                                                      center_y=0))

            agents_dict = self._visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                            color="black", track_id=-2,
                                                                                            agent_type="", prob=0.))

            lanes_dict = self._visualization_list_to_dict(frame.lanes, LaneVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                         color="black", id="none"))

            crosswalk_dict = self._visualization_list_to_dict(frame.crosswalks, CWVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                                color="black"))
            tl_dict = self._visualization_list_to_dict(frame.trafficLights, TLVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                            center_x=0, center_y=0,
                                                                                            state_color="grey", state="unknown",
                                                                                            traffic_light_id="none", face_ids=np.empty(0),
                                                                                            corresponding_lanes=["unknown"]))

            lane_tl_dict = self._visualization_list_to_dict(
                frame.lane_tl_relations, LaneTLRelationVisualization(xs=np.zeros((2,), dtype=float), ys=np.zeros((2,), dtype=float)))

            node_dict = self._visualization_list_to_dict(
                frame.nodes, NodeVisualization(x=0., y=0., id="none"))

            # for trajectory we extract the labels so that we can show them in the legend
            trajectory_dict: Dict[str, Dict[str, Any]] = {}
            for trajectory_label in trajectories_labels:
                trajectories = [
                    el for el in frame.trajectories if el.legend_label == trajectory_label]
                trajectory_dict[trajectory_label] = self._visualization_list_to_dict(trajectories,
                                                                                     TrajectoryVisualization(xs=np.empty(0),
                                                                                                             ys=np.empty(
                                                                                                                 0),
                                                                                                             color="black",
                                                                                                             legend_label="none",
                                                                                                             track_id=-2))

            frame_dict = dict(ego=ColumnDataSource(ego_dict),
                              agents=ColumnDataSource(agents_dict),
                              lanes=ColumnDataSource(lanes_dict),
                              crosswalks=ColumnDataSource(crosswalk_dict),
                              traffic_lights=ColumnDataSource(tl_dict),
                              lane_tl_relations=ColumnDataSource(lane_tl_dict),
                              nodes=ColumnDataSource(node_dict))
            frame_dict.update({k: ColumnDataSource(v)
                               for k, v in trajectory_dict.items()})

            out.append(frame_dict)

        scene_fig = bokeh.plotting.figure(
            title="Scene {}".format(scene_index),
            match_aspect=True,
            x_range=(out[0]["ego"].data["center_x"][0] - self.radius,
                     out[0]["ego"].data["center_x"][0] + self.radius),
            y_range=(out[0]["ego"].data["center_y"][0] - self.radius,
                     out[0]["ego"].data["center_y"][0] + self.radius),
            tools=["pan", "wheel_zoom",
                   agent_hover,
                   tl_hover,
                   lane_hover,
                   "save", "reset"],
            active_scroll="wheel_zoom",
            align='center'
        )

        scene_fig.xgrid.grid_line_color = None
        scene_fig.ygrid.grid_line_color = None

        scene_fig.patches(line_width=0, alpha=0.5, name="lanes",
                          color="color", source=out[0]["lanes"])
        scene_fig.patches(line_width=0, alpha=0.5, color="#B5B50D",
                          source=out[0]["crosswalks"])
        scene_fig.patches(line_width=2, color="#B53331", source=out[0]["ego"])
        scene_fig.patches(line_width=2, color="color",
                          name="agents", source=out[0]["agents"])

        # visualize tls state
        scene_fig.ellipse(x="center_x", y="center_y", width=2, height=2, line_width=2, color="state_color",
                          source=out[0]["traffic_lights"], name="trafficLights")

        # visualize relation

        scene_fig.multi_line(xs="xs", ys="ys",
                             line_width=2, color="#8073ac", source=out[0]["lane_tl_relations"], legend_label="lane_tl_relations")
        # visualize tls geometry
        scene_fig.patches(line_width=2, color="black",
                          source=out[0]["traffic_lights"])

        # # visualize nodes
        # scene_fig.ellipse(x="x", y="y", width=2, height=2, line_width=2, color="black",
        #                   source=out[0]["nodes"], name="nodes", legend_label="nodes")

        js_string = """
                sources["lanes"].data = frames[cb_obj.value]["lanes"].data;
                sources["crosswalks"].data = frames[cb_obj.value]["crosswalks"].data;
                sources["agents"].data = frames[cb_obj.value]["agents"].data;
                sources["ego"].data = frames[cb_obj.value]["ego"].data;
                sources["traffic_lights"].data = frames[cb_obj.value]["traffic_lights"].data;
                sources["lane_tl_relations"].data = frames[cb_obj.value]["lane_tl_relations"].data;
                sources["nodes"].data = frames[cb_obj.value]["nodes"].data;

                var center_x = frames[cb_obj.value]["ego"].data["center_x"][0];
                var center_y = frames[cb_obj.value]["ego"].data["center_y"][0];

                figure.x_range.setv({"start": center_x-100, "end": center_x+100})
                figure.y_range.setv({"start": center_y-100, "end": center_y+100})

                sources["lanes"].change.emit();
                sources["crosswalks"].change.emit();
                sources["agents"].change.emit();
                sources["ego"].change.emit();
                sources["traffic_lights"].change.emit();
                sources["lane_tl_relations"].change.emit();
                sources["nodes"].change.emit();
            """

        for trajectory_name in trajectories_labels:
            scene_fig.multi_line(alpha=0.8, line_width=3, source=out[0][trajectory_name], color="color",
                                 legend_label=trajectory_name)
            js_string += f'sources["{trajectory_name}"].data = frames[cb_obj.value]["{trajectory_name}"].data;\n' \
                f'sources["{trajectory_name}"].change.emit();\n'

        slider_callback = CustomJS(
            args=dict(figure=scene_fig, sources=out[0], frames=out),
            code=js_string,
        )

        slider = Slider(start=0, end=len(frames),
                        value=0, step=1, title="frame")
        slider.js_on_change("value", slider_callback)

        scene_fig.legend.location = "top_left"
        scene_fig.legend.click_policy = "hide"

        layout = column(scene_fig, slider)
        return layout

    def get_satellite_view(self):
        # sat_img = _load_satellite_map(self.cfg["satellite_map_key"], self.dm)
        pass

    def _get_frame_trajectories(self, frames: np.ndarray, agents_frames: List[np.ndarray], track_ids: np.ndarray,
                                frame_index: int) -> List[TrajectoryVisualization]:
        """Get trajectories (ego and agents) starting at frame_index.
        Ego's trajectory will be named ego_trajectory while agents' agent_trajectory

        :param frames: all frames from the scene
        :param agents_frames: all agents from the scene as a list of array (one per frame)
        :param track_ids: allowed tracks ids we want to build trajectory for
        :param frame_index: index of the frame (trajectory will start from this frame)
        :return: a list of trajectory for visualisation
        """

        traj_visualisation: List[TrajectoryVisualization] = []
        agent_traj_length = 20
        for track_id in track_ids:
            # TODO this is not really relative (note eye and 0 yaw)
            pos, *_, avail = get_relative_poses(agent_traj_length, frames[frame_index: frame_index + agent_traj_length],
                                                track_id, agents_frames[frame_index: frame_index +
                                                                        agent_traj_length],
                                                np.eye(3), 0)
            traj_visualisation.append(TrajectoryVisualization(xs=pos[avail > 0, 0],
                                                              ys=pos[avail >
                                                                     0, 1],
                                                              color="blue",
                                                              legend_label="agent_trajectory",
                                                              track_id=int(track_id)))

        ego_traj_length = 100
        pos, *_, avail = get_relative_poses(ego_traj_length, frames[frame_index: frame_index + ego_traj_length],
                                            None, agents_frames[frame_index: frame_index +
                                                                ego_traj_length],
                                            np.eye(3), 0)
        traj_visualisation.append(TrajectoryVisualization(xs=pos[avail > 0, 0],
                                                          ys=pos[avail > 0, 1],
                                                          color="red",
                                                          legend_label="ego_trajectory",
                                                          track_id=-1))

        return traj_visualisation
