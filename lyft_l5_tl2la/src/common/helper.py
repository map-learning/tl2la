# ------------------------------------------------------------------------
# Helper functions of any kind
# Lyft Lvl 5 Dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

import itertools
import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Dict, Tuple, TypeVar
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.filter import filter_tl_faces_by_frames
from scipy.spatial import distance
from src.common.definitions import DATASET_SIZE


DATASET_CONFIG = {
    "dataset_root_path": "./dataset/",
    "intro": "./configs/intro_config.yaml",
    "train1": "./configs/train1_config.yaml",
    "trainfull": "./configs/trainfull_config.yaml",
    "validate": "./configs/validate_config.yaml"
}


def init_dataset(config_name: str) -> Tuple[dict, LocalDataManager, ChunkedDataset]:
    """Initialize zarr_dataset, Local Datamanager and config

    Avaialabel Config names:
        * intro
        * train1
        * trainfull
        * validate

    Args:
        config_name (str): Config name

    Returns:
        Tuple[dict, LocalDataManager, ChunkedDataset]: config, Data Manager, Zarr Dataset 
    """
    os.environ["L5KIT_DATA_FOLDER"] = DATASET_CONFIG["dataset_root_path"]

    cfg = load_config_data(f"./{DATASET_CONFIG[config_name]}")

    dm = LocalDataManager()
    dataset_path = dm.require(cfg["val_data_loader"]["key"])
    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()
    return (cfg, dm, zarr_dataset)


def load_pickled_df(path: str) -> pd.DataFrame:
    """Loads a pickled Dataframe

    Args:
        path (str): path to pkl-file
    Returns:
        pd.DataFrame: unpickeled DataFrame
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_df(path: str, df: pd.DataFrame):
    """Save a Dataframe as pkl-file

    Args:
        path (str): store pkl-file to path
    """
    with open(path, 'wb') as f:
        pickle.dump(df, f)


T = TypeVar('T')


def load_from_pickled_file(path: str) -> T:
    """Loads a pickled file

    Args:
        path (str): path to pkl-file
    Returns:
        T: loaded object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_as_pickled_file(path: str, obj: T):
    """Save to pickled file

    Args:
        path (str): store pkl-file to path
        obj (T): object to pickle
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_filenames(folder_path: str) -> list:
    """Gets the filenames in a specific folder

    Args:
        folder_path (str): path to folder

    Returns:
        list: filenames of files in folder
    """
    f_names = []
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        f_names.extend(filenames)
        break
    f_names = sorted(f_names, key=lambda x: int(
        re.findall(r'\d+', x)[0]), reverse=False)
    return f_names


def concat_preprocessed_dataframes(path: str, start_idx: int = 0, end_idx: int = DATASET_SIZE) -> pd.DataFrame:
    """Concatenate preprocessed dataframes

    Args:
        path (str): dataset path
        start_idx (int, optional): start index. Defaults to 0.
        end_idx (int, optional): end index. Defaults to DATASET_SIZE.

    Returns:
        pd.DataFrame: concatenated dataframe
    """
    dfs = []
    files = get_filenames(path)

    for fname in files[start_idx:end_idx]:
        df = load_pickled_df(path+fname)
        dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def tl_face_status_is_same(face_a: np.ndarray, face_b: np.ndarray):
    """Compares the traffic light face states for two given tl faces.

    Args:
        face_states (np.ndarray): two traffic 

    Returns:
        (bool): depending if states are same or not
    """
    return np.array_equal(face_a, face_b)


def count_conflicting_tl_face_conditions(tl_faces: np.ndarray) -> int:
    """Counts conflicting traffic light faces for a a given number of traffic light faces

    Args:
        tl_faces (np.ndarray): given traffic light faces

    Returns:
        int: number of conflicting tl_faces
    """
    tl_ids: set = set(tl_faces["traffic_light_id"])
    conflicts = 0

    # iterate over traffic lights
    for tl_id in tl_ids:

        # filter for specific traffic light
        current_tl_faces = tl_faces[tl_faces["traffic_light_id"] == tl_id]
        current_face_ids = set(current_tl_faces["face_id"])

        for face_id in current_face_ids:
            same_tl_faces = current_tl_faces[current_tl_faces["face_id"] == face_id]
            same_tl_face_idx = []
            for a_idx, b_idx in itertools.combinations(range(len(same_tl_faces)), 2):
                if not tl_face_status_is_same(same_tl_faces[a_idx]["traffic_light_face_status"], same_tl_faces[b_idx]["traffic_light_face_status"]):
                    # conflicts += 2
                    same_tl_face_idx.extend([a_idx, b_idx])

            # calc number of conflicts for current tl-face
            conflicts += len(same_tl_face_idx)
    return conflicts


def get_specific_tl_faces(dataset: ChunkedDataset, scene_idx: int, frame_idx: int) -> np.ndarray:
    """Get the traffic light faces for a specific frame of a scene

    Args:
        scene_idx (int): scene which contains specific frame
        frame_idx (int): frame of scene

    Returns:
        np.ndarray: tl faces in specific frame
    """

    current_scene = dataset.get_scene_dataset(scene_index=scene_idx)
    frames = current_scene.frames
    tls_frames = filter_tl_faces_by_frames(frames, current_scene.tl_faces)
    return tls_frames[frame_idx]


def get_closest_point(point: np.ndarray, comparison_points: list) -> np.ndarray:
    """Get the closest coparison point to another point

    Args:
        point (np.ndarray): origin point
        comparison_points (list): list of comparison points

    Returns:
        np.ndarray: closest comparison point
    """

    dist_list = []

    for p in comparison_points:
        dist = distance.euclidean(point, p)
        dist_list.append(dist)

    min_dist_index = dist_list.index(min(dist_list))
    return comparison_points[min_dist_index]


def indices_in_bound(center: np.ndarray, elements_center: np.ndarray, half_extent: float) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)

    (element_x - center_x)² + (element_y - center_y)² < half_extent²

    Args:
        center (float): XY of the center
        elements_center (np.ndarray): array of shape Nx2 [[x1,y1], ...,[xn,yn]]
        half_extent (float): half the side of the bounding box centered around center

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    center_x, center_y = center

    mask = (elements_center[:, 0] - center_x)**2 + \
        (elements_center[:, 1] - center_y)**2 <= half_extent**2
    return np.nonzero(mask)


def calculate_center(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
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


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> np.float:
    """Calculates eucledian distance between to given points

    Args:
        p1 (np.ndarray): coordinates point 1
        p2 (np.ndarray): coordinates point 2

    Returns:
        np.float: distance
    """
    return np.linalg.norm(p2-p1)


def filter_unique_labels(ax: plt.Axes) -> Tuple:
    """make sure only unique labels show up in matplotlib legend

    Args:
        ax (plt.Axes): plot axes

    Returns:
        Tuple: handles and labels
    """
    handles, labels = ax.get_legend_handles_labels()
    i = np.arange(len(labels))
    filter = np.array([])  # set up a filter

    unique_labels = list(set(labels))
    for ul in unique_labels:
        # find the first instance of this label and add its index to the filter
        filter = np.append(filter, [i[np.array(labels) == ul][0]])

    handles = [handles[int(f)] for f in filter]
    labels = [labels[int(f)] for f in filter]
    return handles, labels


def get_pickled_dataset_index(scene_idx: int, cfg: dict) -> int:
    """Get index for traffic light and agent mapping dataset 
    that needs to be loaded for a specific scene.

    Args:
        scene_idx (int): index of a scene
        cfg (dict): dataset config

    Returns:
        int: common index of pickle file
    """
    DATASET_FILES = 1000
    tl_scenes_path = cfg["preprocessed_data_loader"]["tls_scene_indices"]
    with open(tl_scenes_path, 'rb') as f:
        tls_scene_indices = pickle.load(f)

    splitted_scenes = np.array_split(tls_scene_indices, DATASET_FILES)
    for idx, group in enumerate(splitted_scenes):
        if scene_idx >= group[0] and scene_idx <= group[-1]:
            return idx

    raise ValueError(f"Index for scene {scene_idx} not found!")


def reverse_dict_of_sets(dict_to_reverse: Dict[T, set]) -> Dict[T, set]:
    """Revese key value pairs of dictionary that stand in a specific relation

    Args:
        dict_to_reverse (Dict["str", set]): dictonary of sets that needs to be reversed

    Returns:
        Dict["str", set]: reversed dict
    """
    reversed_dict = dict()
    for lane_id, tl_set in dict_to_reverse.items():
        for tl_id in tl_set:
            if tl_id not in reversed_dict.keys():
                reversed_dict[tl_id] = {lane_id}
            else:
                reversed_dict[tl_id] = reversed_dict[tl_id].union({
                                                                  lane_id})

    return reversed_dict
