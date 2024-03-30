# ------------------------------------------------------------------------
# Defines object that handles all operation of transformed Lyft Lvl 5 
# dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from l5kit.data import ChunkedDataset
from l5kit.data.filter import (filter_agents_by_distance,
                               filter_agents_by_frames,
                               filter_tl_faces_by_frames)
from l5kit.data.labels import TL_FACE_LABEL_TO_INDEX
from l5kit.data.filter import filter_agents_by_labels
from loguru import logger
from src.dataset_operations.branch_dataset import BranchDataset
from src.common.definitions import DATASET_TYPE, TL_STATUS
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.helper import load_pickled_df, reverse_dict_of_sets
from src.common.utils import map_agent_to_lanes
from torch.utils.data import Dataset
from tqdm import tqdm


class JunctionDataset(Dataset):
    """Junction dataset that is used for all dataset transformations based on the Lyft dataset
    """

    def __init__(
            self,
            cfg: dict,
            mapAPI: ExtendedMapAPI,
            zarr_dataset: ChunkedDataset,
    ):
        """
        Initalize junction dataset that is used for all dataset transformations based on the Lyft dataset

        Args:
            cfg (dict): configuration file
            mapAPI (ExtendedMapAPI): get information about static environment
            zarr_dataset (ChunkedDataset): the raw zarr datasetx
        """
        self.cfg = cfg
        self.mapAPI = mapAPI
        self.dataset = zarr_dataset
        self.branch_dataset = BranchDataset()

        # List of traffic light controlled junctions
        self.tl_junction_ids = self.get_list_of_junctions(
            tl_junctions_only=True)
        
        self._lane_tl_relations: Optional[dict] = None
        self._tl_lane_relations: Optional[dict] = None
        self._ego_lane_mapping: Optional[pd.DataFrame] = None
        self._ego_branch_mapping: Optional[pd.DataFrame] = None
        self._tl_scene_indices: Optional[list] = None

        self._cached_agent_mapping: Optional[pd.DataFrame] = None
        
    @property
    def lane_tl_relations(self) -> Dict[str, set]:
        """List for all lane ids the corresponding traffic lights

        Returns:
            Dict[str, set]: Lane id -> set of corresponding traffic light ids
        """

        if self._lane_tl_relations is None:
            self._lane_tl_relations = self.mapAPI.get_lane_tl_relation()

        return self._lane_tl_relations

    @property
    def tl_lane_relations(self) -> Dict[str, set]:
        """List for all traffic lights the corresponding lanes

        Returns:
            Dict[str, set]: Traffic light id -> set of corresponding lane ids
        """
        return reverse_dict_of_sets(self.lane_tl_relations)
    
    @property
    def frame_lanes_df(self) -> pd.DataFrame:
        """Mapping of ego vehicle position to lane ids (1:n) for all frames

        Returns:
            pd.DataFrame: scene_idx, frame_idx, agent_id, lane_ids
        """
        if self._ego_lane_mapping is None:
            self._ego_lane_mapping = load_pickled_df(self.cfg["preprocessed_data_loader"]["ego_frame_lane_mapping"])
        return self._ego_lane_mapping
    
    @property
    def frame_branch_id_df(self) -> pd.DataFrame:
        """Mapping of ego vehicle position to branch id (1:0..1) for all frames

        Returns:
            pd.DataFrame: scene_idx, frame_idx, agent_id, lane_ids, branch_id, junctionm_id
        """
        if self._ego_branch_mapping is None:
            self._ego_branch_mapping = self.get_frame_branch_id_df()
        return self._ego_branch_mapping
    
    @property
    def tls_scene_indices(self) -> list:
        """Mapping of ego vehicle position to lane ids (1:n) for all frames

        Returns:
            list: traffic light scene indices
        """
        if self._tl_scene_indices is None:
            self._tl_scene_indices = self.frame_lanes_df["scene_idx"].unique().tolist()
        return self._tl_scene_indices
    
    def __len__(self) -> int:
        """
        Get the number of available frames

        Returns:
            int: the number of frames in the dataset
        """
        return len(self.dataset.frames)

    def __getitem__(self, idx: int) -> int:
        """ Get a traffic light scene for a given frame index
        """
        # TODO: implement (not used)
        return 0

    def get_frame_branch_id_df(self) -> pd.DataFrame:
        branches_df = self.get_branch_dataset(DATASET_TYPE.EXTENDED)
        merged_df = self.frame_lanes_df.join(
            branches_df.set_index('lane_id'), on=['lane_id'])
        merged_df = merged_df.dropna()
        return merged_df

    def get_list_of_junctions(self, tl_junctions_only=False) -> List["str"]:
        """Lists all junctions of semantic map.
        If filter tl_junctions_only is set, then only junctions with traffic lights / lane relations are shown.

        Args:
            tl_junctions_only (bool): filter for junctions with traffic lights

        Returns:
            List["str"]: ids of junctions
        """

        # List of self labeled traffic light controlled junctions
        TL_JUNCTION_IDS = ["LrDx", "y4Ss", "3Ccg", "E0Us",
                           "U6Ym", "FToR", "ECHt", "DoQH", "ctKJ", "qnJy", "RYcT"]

        if tl_junctions_only:
            return TL_JUNCTION_IDS
        return self.mapAPI.get_element_ids_from_layer("junction")

    def append_dict_values(self, dict_to_append: dict, key: str, value: Union[str, int]) -> dict:
        """Append dictionary value to existing or non existing key

        Args:
            dict_to_append (dict): dictionary to append value
            key (str): key to apend value
            value (Union[str, int]): the value

        Returns:
            dict: appended dictionary
        """
        if key not in dict_to_append.keys():
            dict_to_append[key] = {value}
        else:
            dict_to_append[key] = dict_to_append[key].union({value})

        return dict_to_append

    def get_minimal_tl_lane_relations(self) -> dict:
        """Reduce tl to lane segment relations to the segment head lane id

        Returns:
            dict: minimal tl to lane relations
        """

        # load lane to tl relations lookup if not already loaded
        if self.lane_tl_relations is None:
            self.lane_tl_relations = self.mapAPI.get_lane_tl_relation()

        # load lane to tl relations lookup if not already loaded
        if self.lane_tl_relations is None:
            self.lane_tl_relations = self.mapAPI.get_lane_tl_relation()

        reduced_tl_lane_dict: Dict[str, set] = {}
        for tl_id in self.tl_lane_relations.keys():
            lane_ids = self.tl_lane_relations[tl_id]
            if lane_ids is None:
                continue
            reduced_tl_lane_dict[tl_id] = set()
            for lane_id in lane_ids:
                predecessor_lanes = self.mapAPI.get_lane_predecessors(lane_id)
                has_predecessor = any(predecessor in self.lane_tl_relations.keys()
                                      for predecessor in predecessor_lanes)
                if not has_predecessor:  # identify lane segment head
                    reduced_tl_lane_dict[tl_id] = reduced_tl_lane_dict[tl_id].union(
                        set([lane_id]))
            if not reduced_tl_lane_dict[tl_id]:
                del reduced_tl_lane_dict[tl_id]
        return reduced_tl_lane_dict

    def get_tl_lane_relations(self) -> pd.DataFrame:
        """Reduce tl to lane segment relations to the segment head lane id

        Returns:
            pd.DataFrame: minimal tl to lane relations
        """

        # load lane to tl relations lookup if not already loaded
        if self.lane_tl_relations is None:
            self.lane_tl_relations = self.mapAPI.get_lane_tl_relation()

        relations = []
        minimal_lane_tl_relations = reverse_dict_of_sets(
            self.get_minimal_tl_lane_relations())
        branches_df = self.get_branch_dataset(DATASET_TYPE.EXTENDED)
        for _, row in branches_df.iterrows():
            current_lane_id = row["lane_id"]
            if current_lane_id in minimal_lane_tl_relations.keys():
                for tl_id in self.lane_tl_relations[current_lane_id]:
                    relations.append(
                        [row["junction_id"], row["branch_id"], tl_id, current_lane_id])

        tl_lane_relation = pd.DataFrame(
            relations, columns=["junction_id", "branch_id", "tl_id", "lane_id"])
        return tl_lane_relation

    def get_classification_classes(self) -> pd.DataFrame:
        """Get tl to lane relations that need to be classified.

        Returns:
            pd.DataFrame: Tl to lane relations
        """
        path = self.cfg["preprocessed_data_loader"]["ground_truth"]
        target_df = load_pickled_df(path)
        return target_df.drop("relation", axis=1)

    def get_target_relations(self) -> pd.DataFrame:
        """Get target traffic light to lane associations, that need to be classified

        Returns:
            pd.DataFrame: target classes
        """

        path = self.cfg["preprocessed_data_loader"]["ground_truth"]
        return load_pickled_df(path).drop("relation", axis=1)

    def get_target_relations_corrected(self) -> pd.DataFrame:
        """Get target traffic light to lane associations, that need to be classified

        Returns:
            pd.DataFrame: target classes
        """

        path = self.cfg["preprocessed_data_loader"]["ground_truth_corrected"]
        return pd.read_csv(path).drop(["relation", "corrected"], axis=1)

    def get_ground_truth(self) -> pd.DataFrame:
        """Get ground trouth labels for traffic light to lane association

        Returns:
            pd.DataFrame:ground trouth label
        """

        path = self.cfg["preprocessed_data_loader"]["ground_truth"]
        return load_pickled_df(path)

    def get_ground_truth_corrected(self) -> pd.DataFrame:
        """Get manually corrected ground trouth labels for traffic light to lane association 

        Returns:
            pd.DataFrame:ground trouth label
        """

        path = self.cfg["preprocessed_data_loader"]["ground_truth_corrected"]
        return pd.read_csv(path)

    def get_junction_lane_relation(self) -> Dict["str", set]:
        """Get the relation of a junction and its corresponding lanes for
        a given semantic map.

        Returns:
            Dict["str", set]: junct_id -> set lane ids
        """

        junction_ids: list = self.mapAPI.get_element_ids_from_layer("junction")
        junction_lanes_dict = dict()

        for junction_id in junction_ids:
            current_junction = self.mapAPI.get_element_by_id(junction_id)
            lane_ids_of_junction = [self.mapAPI.id_as_str(
                lane_id) for lane_id in current_junction.element.junction.lanes]

            if len(lane_ids_of_junction) != 0:
                junction_lanes_dict[junction_id] = set(lane_ids_of_junction)

        return junction_lanes_dict

    def reduce_dict(self, dict_to_reduce: dict, existing_keys: list) -> dict:
        """Reduce a dictionary to the list of existing keys, if they exist in the original dict

        Args:
            dict_to_reduce (dict): dictonary to reduce
            existing_keys (list): keys that exist in the final reduced dictionary, if they existed in the original dict

        Returns:
            dict: reduced dictionary
        """
        subtracted_keys = set(existing_keys) - set(dict_to_reduce.keys())
        if len(subtracted_keys) != 0:
            raise ValueError(
                f"Some keys dont exist in original dictionary: {subtracted_keys}")

        reduced_dict = {k: v for k, v in dict_to_reduce.items()
                        if k in existing_keys}

        return reduced_dict

    def get_frame_lane_ids(self, scene_indices: list) -> pd.DataFrame:
        """Get the lane_id on which the AV is located for each frame:
        Dataframe Columns:
        * scene_idx (int)
        * frame_idx (int)
        * agent_id (int)
        * lane_ids (List[str])

        Returns:
            pd.DataFrame: frame_lane_df
        """

        frame_lane_ids = []

        for scene_idx in tqdm(scene_indices, desc="Iterate through scenes to filter scenes by junctions"):
            current_scene = self.dataset.get_scene_dataset(
                scene_index=scene_idx)
            frames = current_scene.frames
            agents_frames = filter_agents_by_frames(
                frames, current_scene.agents)

            for frame_idx in range(len(frames)):
                ego_frame = frames[frame_idx]
                agents_frame = agents_frames[frame_idx]
                agents_frame = filter_agents_by_labels(agents_frame, 0.9)
                agents_frame = filter_agents_by_distance(
                    agents_frame, ego_frame["ego_translation"][:2], 60)
                current_rows = []

                # ego mapping
                lane_ids = map_agent_to_lanes(
                    ego_frame, self.mapAPI, scene_idx, frame_idx, is_ego=True)
                ego_mapping = [scene_idx, frame_idx, 0, lane_ids]
                current_rows.append(ego_mapping)

                frame_lane_ids.extend(current_rows)
        df = pd.DataFrame(frame_lane_ids, columns=["scene_idx", "frame_idx", "agent_id", "lane_ids"])
        return df

    def map_agents_to_lanes(self, scene_idx: int) -> List:
        """Get the lane_id on which the AV is located for each frame
        Dataframe Columns:
        * scene_idx (int)
        * frame_idx (int)
        * agent_id (int)
        * lane_ids (List[str])

        Returns:
            list: frame_lane_ids
        """
        logger.info(f"Start mapping for Scene {scene_idx}.")

        frame_lane_ids = []
        current_scene = self.dataset.get_scene_dataset(scene_index=scene_idx)
        logger.info(f"Fetched Scene {scene_idx}.")

        frames = current_scene.frames
        agents_frames = filter_agents_by_frames(
            frames, current_scene.agents)

        for frame_idx in range(len(frames)):
            ego_frame = frames[frame_idx]
            agents_frame = agents_frames[frame_idx]
            agents_frame = filter_agents_by_labels(agents_frame, 0.9)
            agents_frame = filter_agents_by_distance(
                agents_frame, ego_frame["ego_translation"][:2], 60)
            current_rows = []

            logger.debug(
                f"Mapping ego in Scene {scene_idx} and frame {frame_idx}.")
            # ego mapping
            lane_ids = map_agent_to_lanes(
                ego_frame, self.mapAPI, scene_idx, frame_idx, is_ego=True)
            ego_mapping = [scene_idx, frame_idx, 0, lane_ids]
            current_rows.append(ego_mapping)

            # mapping for all agents
            for agent in agents_frame:
                lane_ids = map_agent_to_lanes(
                    agent, self.mapAPI, scene_idx, frame_idx, is_ego=False)
                agent_mapping = [scene_idx, frame_idx,
                                 agent['track_id'], lane_ids]
                current_rows.append(agent_mapping)

                frame_lane_ids.extend(current_rows)

        logger.info(f"Mapping for Scene {scene_idx} done.")
        return frame_lane_ids

    def map_tl_status(self, scene_idx: int) -> List:
        """Get the lane_id on which the AV is located for each frame
        Dataframe Columns:
        * scene_idx (int)
        * frame_idx (int)
        * tl_id (str)
        * tl_status (str)
        * active_faces (list)
        * inactive_faces (list)
        * unknown_faces (list)

        Returns:
            list: frame_tl_ids
        """
        logger.info(f"Start mapping for Scene {scene_idx}.")

        frame_tl_ids = []
        current_scene = self.dataset.get_scene_dataset(scene_index=scene_idx)
        logger.info(f"Fetched Scene {scene_idx}.")

        frames = current_scene.frames
        tls_frames = filter_tl_faces_by_frames(frames, current_scene.tl_faces)

        for frame_idx in range(len(frames)):
            current_rows = []
            current_traffic_light_ids = np.unique(
                tls_frames[frame_idx]["traffic_light_id"]).tolist()
            for current_tl_id in current_traffic_light_ids:
                current_tl_face_ids = np.unique(
                    tls_frames[frame_idx][tls_frames[frame_idx]["traffic_light_id"] == current_tl_id])

                active_tfl_face_ids = self.filter_tl_face_ids_by_status(
                    current_tl_face_ids, "ACTIVE")
                inactive_tfl_faces = self.filter_tl_face_ids_by_status(
                    current_tl_face_ids, "INACTIVE")
                unknown_tfl_faces = self.filter_tl_face_ids_by_status(
                    current_tl_face_ids, "UNKNOWN")
                tl_status = self.get_tl_status(active_tfl_face_ids)
                current_row = [scene_idx, frame_idx, current_tl_id, tl_status,
                               active_tfl_face_ids, inactive_tfl_faces, unknown_tfl_faces]
                current_rows.append(current_row)

            frame_tl_ids.extend(current_rows)

        logger.info(f"Mapping for Scene {scene_idx} done.")
        return frame_tl_ids

    def get_tl_status(self, active_face_ids: np.ndarray) -> int:
        """Returns the status of the faces of a given traffic light

        Args:
            active_faces (np.ndarray): active traffic light faces

        Returns:
            int: status of enmum: TL_STATUS

        TL_STATUS(enum):
            * RED = 0
            * GREEN = 1
            * YELLOW = 2
            * YELLOW_RED = 3
            * UNKNOWN = 4
        """
        face_colors = []
        for active_face_id in active_face_ids:
            face_colors.append(self.mapAPI.get_color_for_face(active_face_id))

        if len(face_colors) == 1:
            return TL_STATUS[face_colors[0]].value
        elif len(face_colors) == 2:
            states = [TL_STATUS[face_colors[0]].value,
                      TL_STATUS[face_colors[1]].value]
            if sum(states) == 2:  # red + yellow
                return 3

        return 4  # all other states that might be undefined

    def filter_tl_face_ids_by_status(self, face_ids: np.ndarray, tl_face_status: str) -> np.ndarray:
        """Filter traffic light faces by status

        Args:
            face_ids (np.ndarray): array of traffic light face ids
            tl_face_status (str): status to filter for

        Returns:
            face_ids: face ids with given status
        """
        face_idx = TL_FACE_LABEL_TO_INDEX[tl_face_status]
        return face_ids[face_ids["traffic_light_face_status"][:, face_idx] == 1.]["face_id"]

    def get_scene_indices_for_junction(self, junction_id: str) -> List[int]:
        """Get the scene indices for a junction which an ego vehicle recorded

        Args:
            junction_id (str): Id of junction

        Returns:
            List[int]: scene indices for a junction which an ego vehicle recorded
        """
        return self.frame_branch_id_df[self.frame_branch_id_df["junction_id"] == junction_id]["scene_idx"].unique().tolist()

    def get_scene_indices_for_branch(self, branch_id: str) -> List[int]:
        """Get the scene indices for a junction branch which an ego vehicle recorded

        Args:
            branch_id (str): Id of junction branch

        Returns:
            List[int]: scene indices for a junction branch which an ego vehicle recorded
        """
        return self.frame_branch_id_df[self.frame_branch_id_df["branch_id"] == branch_id]["scene_idx"].unique().tolist()

    def get_branch_ids_of_junction(self, junction_id: str) -> List["str"]:
        """Get the branch id of a junction

        Args:
            junction_id (str): junction id

        Returns:
            list: branch ids of junction
        """
        branches_df = self.get_branch_dataset(DATASET_TYPE.EXTENDED)
        return branches_df[branches_df["junction_id"] == junction_id]["branch_id"].unique().tolist()

    def __get_index(self, splitted_array: list, value: int) -> int:
        """get the index of an element

        Args:
            splitted_array (list): splitted array
            value (int): value to search for

        Returns:
            int: return index of outter array in list
        """
        for idx, arr in enumerate(splitted_array):
            if value < arr[-1]:
                return idx
        raise ValueError(f"Index {value} not in traffic light scenes!")

    
    def get_agent_mapping_for_scene(self, scene_idx: int) -> pd.DataFrame:
        """Get the agent to lane relation for each frame in a scene

        Args:
            scene_idx (int): scene id

        Returns:
            pd.DataFrame: agent mapping in scene
        """

        # check if scene already in cached scenes
        if (self._cached_agent_mapping is not None) and (scene_idx in self._cached_agent_mapping["scene_idx"].unique().tolist()):
            return self._cached_agent_mapping[self._cached_agent_mapping["scene_idx"] == scene_idx]

        # otherwise: get agent mapping df
        splitted_scenes = np.array_split(self.tls_scene_indices, 1000)

        df_idx = self.__get_index(splitted_scenes, scene_idx)
        self.cached_agent_mapping = load_pickled_df(os.path.join(
            self.cfg["preprocessed_data_loader"]["agent_frame_lane_mapping_folder"], f"{df_idx}_df_trainfull.pkl"))

        return self.cached_agent_mapping[self.cached_agent_mapping["scene_idx"] == scene_idx]


    def get_tl_of_junction(self, junction_id: str) -> list:
        """Get a list of traffic light ids of a specific junction

        Args:
            junction_id (str): junction id to get traffic lights from

        Returns:
            list: list of corresponding traffic light ids
        """
        tls_of_junction: List[str] = []
        branches_df = self.get_branch_dataset(DATASET_TYPE.EXTENDED)
        lanes_of_junction = branches_df[branches_df["junction_id"]
                                        == junction_id]["lane_id"].tolist()
        for lane_id in lanes_of_junction:
            if lane_id in self.lane_tl_relations.keys():
                tls_of_junction.extend(self.lane_tl_relations[lane_id])
        return list(set(tls_of_junction))

    def get_agent_mapping(self, index: int) -> pd.DataFrame:
        """Loads the preprocessed agent to lane mapping dataset from file system

        Args:
            index (int): dataset index to load

        Returns:
            pd.DataFrame: loaded dataset
        """
        path = os.path.join(self.cfg["preprocessed_data_loader"]["agent_frame_lane_mapping_folder"], f"{index}_df_trainfull.pkl")
        return load_pickled_df(path)

    def get_agent_motion(self, index: int) -> pd.DataFrame:
        """Loads the preprocessed agent motion dataset from file system

        Args:
            index (int): dataset index to load

        Returns:
            pd.DataFrame: loaded dataset
        """
        path = os.path.join(self.cfg["preprocessed_data_loader"]["agent_motion"], f"{index}_df_trainfull.pkl")
        return load_pickled_df(path)

    def get_scene_tl_state(self) -> pd.DataFrame:
        """Loads the preprocessed scene tl_state dataset from file system

        Args:
            index (int): dataset index to load

        Returns:
            pd.DataFrame: loaded dataset
        """
        path = self.cfg["preprocessed_data_loader"]["scene_tl_state"]
        return load_pickled_df(path)

    def get_tl_mapping(self, index: int) -> pd.DataFrame:
        """Loads the preprocessed tl to lane mapping dataset from file system

        Args:
            index (int): dataset index to load

        Returns:
            pd.DataFrame: loaded dataset
        """
        path = f"./dataset/preprocessed_info/tl_mapping/{index}_df_trainfull.pkl"
        return load_pickled_df(path)

    def get_combined_features(self, index: int) -> pd.DataFrame:
        """Loads the preprocessed dataset with agent to lane mapping and current tl states from file system

        Args:
            index (int): dataset index to load

        Returns:
            pd.DataFrame: loaded dataset
        """
        path = f"./dataset/preprocessed_info/combined_features/{index}_df_trainfull.pkl"
        return load_pickled_df(path)

    def get_branch_dataset(self, d_type: DATASET_TYPE) -> pd.DataFrame:
        """Loads specific branch dataset from file system


        Args:
            type (int): dataset index to load

        Returns:
            pd.DataFrame: loaded dataset
        """
        return self.branch_dataset.get_branch_dataset(d_type)

    def get_branch_of_lane(self, lane_id: str) -> Optional[str]:
        """Return branch id for a given lane segment

        Args:
            lane_id (str): lane segment id

        Returns:
            str: id of branch
        """
        branches_df = self.get_branch_dataset(DATASET_TYPE.EXTENDED)
        branch_of_lane = branches_df[branches_df["lane_id"] == lane_id].get(
            "branch_id", [])
        if len(branch_of_lane) == 0:
            return None
        return branch_of_lane.iloc[0]

    def get_junction_of_branch(self, branch_id: str) -> str:
        """Get the junction id for a given branch

        Args:
            branch_id (str): branch id

        Returns:
            str: junction id
        """
        branches_df = self.get_branch_dataset(DATASET_TYPE.EXTENDED)
        if branch_id not in branches_df["branch_id"].unique().tolist():
            raise ValueError(f"Given branch id: {branch_id} does not belong to a junction!")
        junction = branches_df[branches_df["branch_id"]
                               == branch_id]["junction_id"].unique().tolist()
        return junction[0]  # first and only element

    def get_sequence_head_lane_id(self, lane_id: str) -> str:
        """Return sequence head lane id for a given lane segment

        Args:
            lane_id (str): lane segment id

        Returns:
            str: id of lane segment with the same lane_sequence_id
        """
        branches_df = self.get_branch_dataset(DATASET_TYPE.EXTENDED)
        head_of_sequence = branches_df[branches_df["lane_id"] == lane_id].get(
            "sequence_head_lane_id", [])
        if len(head_of_sequence) == 0:
            return None
        return head_of_sequence.iloc[0]

    def concat_chuncked_datasets(self, d1: ChunkedDataset, d2: ChunkedDataset) -> ChunkedDataset:
        """Concats tow chuncked datasets

        Args:
            d1 (ChunkedDataset): first chuncked dataset
            d2 (ChunkedDataset): second chuncked dataset

        Returns:
            ChunkedDataset: combined chuncked dataset
        """
        d_concat = ChunkedDataset(d1.path, d1.key)
        d_concat.frames = np.concatenate((d1.frames, d2.frames))
        d_concat.scenes = np.concatenate((d1.scenes, d2.scenes))
        d_concat.agents = np.concatenate((d1.agents, d2.agents))
        d_concat.tl_faces = np.concatenate((d1.tl_faces, d2.tl_faces))
        return d_concat
