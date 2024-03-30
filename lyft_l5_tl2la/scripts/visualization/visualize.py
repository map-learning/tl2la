# ------------------------------------------------------------------------
# Run Visualizer to render a single scene of the Lyft Level 5 dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from L5Kit (https://github.com/woven-planet/l5kit)
# Copyright (c) 2022 Woven by Toyota. All Rights Reserved.
# ------------------------------------------------------------------------

from src.visualization.visualizer import Visualizer
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.helper import init_dataset
from bokeh.io import save

# Load Dataset
cfg, dm, zarr_dataset = init_dataset("trainfull")

mapAPI = ExtendedMapAPI.from_config(dm, cfg)

# Adapt Parameters to visualize a different scene or a bigger radius
scene_idx = 2101 
visualizer = Visualizer(cfg, dm, radius=50)

out = visualizer.zarr_to_visualizer_scene(
    zarr_dataset.get_scene_dataset(scene_idx), mapAPI)
out_vis = visualizer.visualize(scene_idx, out)

save(out_vis, f"./res/bokeh/visualization_{scene_idx}.html")

