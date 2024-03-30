# ------------------------------------------------------------------------
# Preprocessing script to map agents to their corresponding lanes
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


import os
import pickle
import re
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from src.common.helper import get_filenames, init_dataset, pickle_df
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.dataset_operations.junction_dataset import JunctionDataset


def get_scene_entry(path):
    """Returns entry index of scene group for processing
    """
    f_names = get_filenames(path)
    if len(f_names) == 0:
        return 0
    # index of last saved scene group +1
    return int(re.findall(r'\d+', f_names[-1])[0]) + 1


@logger.catch
def main():
    # configure logging
    log_file_path: str = os.path.join(
        'logs', 'agent_mapping', 'file_{time}.log')
    logger.add(log_file_path, retention=2,
               encoding="utf8", rotation="500 MB", level="TRACE")  # Debug

    logger.info("Initializing Dataset")
    # dataset_type = "train1"
    dataset_type = "trainfull"
    cfg, dm, zarr_dataset = init_dataset(config_name=dataset_type)

    mapAPI = ExtendedMapAPI.from_config(dm, cfg)
    junctionDataset = JunctionDataset(cfg, mapAPI, zarr_dataset)

    logger.info("Load Traffic Light Scene Indices")
    with open(f'./dataset/preprocessed_info/tls_scene_indices_{dataset_type}.pkl', 'rb') as f:
        tls_scene_indices = pickle.load(f)

    logger.info("Start agent mapping")
    t_start = perf_counter()

    splitted_scenes = np.array_split(tls_scene_indices, 1000)

    path = f'./dataset/preprocessed_info/agent_mapping/'
    entry_idx = get_scene_entry(path)

    for i, scenes in enumerate(splitted_scenes[entry_idx:]):
        results = []
        for scene_idx in scenes:
            # Map Agents in Frames to Lane
            current_results = junctionDataset.map_agents_to_lanes(scene_idx)
            results.extend(current_results)

        df = pd.DataFrame(results, columns=[
            "scene_idx", "frame_idx", "agent_id", "lane_ids"])
        path = f'./dataset/preprocessed_info/agent_mapping/{i}_df_{dataset_type}.pkl'
        pickle_df(path, df)

    t_stop = perf_counter()
    logger.info(f"Elapsed time: {t_stop-t_start}")


if __name__ == "__main__":
    main()
