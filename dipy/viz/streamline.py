from __future__ import division

import numpy as np
from itertools import izip

from dipy.viz import fvtk
from dipy.viz.axycolor import distinguishable_colormap

from dipy.tracking.streamline import get_bounding_box_streamlines


def show_hierarchical_clusters(tree, theta_range=(0, np.pi), show_circles=False, size=(900, 900)):
    bg = (1, 1, 1)
    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)

    box_min, box_max = get_bounding_box_streamlines(tree.root)
    width, height, depth = box_max - box_min
    box_size = max(width, height, depth)

    thresholds = set()
    max_threshold = tree.root.threshold
    box_size *= len(tree.root.children) * (theta_range[1]-theta_range[0]) / (2*np.pi)

    def _draw_subtree(node, color=fvtk.colors.orange_red, theta_range=theta_range, parent_pos=(0, 0, 0)):
        print np.array(theta_range) / np.pi * 360

        # Draw node
        offset = np.zeros(3)
        theta = theta_range[0] + (theta_range[1] - theta_range[0]) / 2.

        radius = max_threshold - node.threshold
        thresholds.add(node.threshold)

        offset[0] += radius*box_size * np.cos(theta)
        offset[1] -= radius*box_size * np.sin(theta)
        fvtk.add(ren, fvtk.line([s + offset for s in node], [color]*len(node), linewidth=2))
        fvtk.add(ren, fvtk.line(np.array([parent_pos, offset]), fvtk.colors.black, linewidth=1))

        if len(node.children) == 0:
            return

        children = sorted(node.children, key=lambda c: len(c))
        ratios = np.maximum([len(c) / len(node) for c in children], 0.1)
        ratios = ratios / np.sum(ratios)  # Renormalize
        sections = theta_range[0] + np.cumsum([0] + ratios.tolist()) * (theta_range[1] - theta_range[0])

        colormap = distinguishable_colormap(bg=bg)
        for i, (node, color) in enumerate(izip(children, colormap)):
            _draw_subtree(node, color, (sections[i], sections[i+1]), offset)

    _draw_subtree(tree.root)

    # Draw circles for the different radius
    if show_circles:
        for threshold in sorted(thresholds)[:-1]:
            radius = max_threshold - threshold
            theta = -np.linspace(*theta_range, num=200)
            X = radius*box_size * np.cos(theta)
            Y = radius*box_size * np.sin(theta)
            Z = np.zeros_like(X)
            dashed_line = zip(np.array([X, Y, Z]).T[::4], np.array([X, Y, Z]).T[1::4])
            fvtk.add(ren, fvtk.line(dashed_line, fvtk.colors.black, linewidth=1))

            scale = box_size/8.
            text = "{:.1f}mm".format(threshold)
            pos = np.array([X[0], Y[0], Z[0]]) + np.array([-len(text)/2.*scale, scale/2., 0])
            fvtk.label(ren, text=text, pos=pos, scale=scale, color=(0, 0, 0))

            pos = np.array([X[-1], Y[-1], Z[-1]]) + np.array([-len(text)/2.*scale, scale/2., 0])
            fvtk.label(ren, text=text, pos=pos, scale=scale, color=(0, 0, 0))

    fvtk.show(ren, size=size)
