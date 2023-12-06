from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from omnivault.linear_algebra.plotter import (
    VectorPlotter2D,
    VectorPlotter3D,
    add_text_annotations,
    add_vectors_to_plotter,
)
from omnivault.linear_algebra.vector import Vector, Vector2D, Vector3D
from omnivault.utils.visualization.figure_manager import FigureManager

# from mpl_toolkits.mplot3d import Axes3D


fig, ax = plt.subplots(figsize=(9, 9))

plotter = VectorPlotter2D(
    fig=fig,
    ax=ax,
    ax_kwargs={
        "set_xlim": {"left": 0, "right": 5},
        "set_ylim": {"bottom": 0, "top": 5},
        "set_xlabel": {"xlabel": "x-axis", "fontsize": 16},
        "set_ylabel": {"ylabel": "y-axis", "fontsize": 16},
        "set_title": {"label": "Vector Magnitude Demonstration", "size": 18},
    },
)

v = Vector2D(
    origin=(0, 0),
    direction=(3, 4),
    color="r",
    label="$\|\mathbf{v}\|_2 = \sqrt{3^2 + 4^2} = 5$",
)
horizontal_component_v = Vector2D(
    origin=(0, 0), direction=(3, 0), color="b", label="$v_1 = 3$"
)
vertical_component_v = Vector2D(
    origin=(3, 0), direction=(0, 4), color="g", label="$v_2 = 4$"
)
add_vectors_to_plotter(plotter, [v, horizontal_component_v, vertical_component_v])
add_text_annotations(plotter, [v])

plotter.plot()
plt.show()
