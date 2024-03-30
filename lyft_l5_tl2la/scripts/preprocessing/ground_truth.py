# ------------------------------------------------------------------------
# Preprocessing of ground truth dataset (afterwards manually corrected 20 
# traffic light labels which are incorrectly labeled by Lyft)
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------


from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.helper import init_dataset, pickle_df
from src.dataset_operations.junction_dataset import JunctionDataset
from src.dataset_operations.preprocessing import generate_target_association


#  Load Dataset
cfg, dm, zarr_dataset = init_dataset("trainfull")

mapAPI = ExtendedMapAPI.from_config(dm, cfg)
junctionDataset = JunctionDataset(cfg, mapAPI, zarr_dataset)

target_df = generate_target_association(mapAPI, junctionDataset)
path = "./dataset/preprocessed_info/target_associations.pkl"
pickle_df(path, target_df)
