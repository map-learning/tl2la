# ------------------------------------------------------------------------
# Run Visualizer to render a static environment of the Lyft Level 5 dataset
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from L5Kit (https://github.com/woven-planet/l5kit)
# Copyright (c) 2022 Woven by Toyota. All Rights Reserved.
# ------------------------------------------------------------------------

from src.visualization.static_visualizer import StaticVisualizer
from src.dataset_operations.extended_map_api import ExtendedMapAPI
from src.common.helper import init_dataset
from bokeh.io import save

# Load Dataset
d_type = "trainfull"
cfg, dm, zarr_dataset = init_dataset(d_type)

mapAPI = ExtendedMapAPI.from_config(dm, cfg)
visualizer = StaticVisualizer(cfg, dm)

out = visualizer.get_semantic_visualisation(mapAPI, heatmap=True)
out_vis = visualizer.visualize(out, heatmap=True)

save(out_vis, f"./res/bokeh/static_visualization_heatmap.html")
