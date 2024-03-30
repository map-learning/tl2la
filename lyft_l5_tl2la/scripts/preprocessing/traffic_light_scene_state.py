# ------------------------------------------------------------------------
# Process script  that assigns a traffic light in a scene a specific state
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

from src.dataset_operations.preprocessing import generate_traffic_light_scene_table
from src.common.helper import pickle_df

save_path = "dataset/preprocessed_info/traffic_light_state_scene_df.pkl"
df = generate_traffic_light_scene_table()
df = df.reset_index(drop=True)
pickle_df(save_path, df)
