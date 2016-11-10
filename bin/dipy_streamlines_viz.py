#! /usr/bin/env python
import os
import numpy as np
from os.path import join as pjoin

import nibabel as nib
from nibabel.streamlines import Tractogram

from dipy.data import read_viz_icons
from dipy.fixes import argparse

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric as MDF

from dipy.viz import window, actor, gui_2d, utils, gui_follower
from dipy.viz.colormap import distinguishable_colormap

from dipy.viz.interactor import CustomInteractorStyle


def build_args_parser():
    description = "Streamlines visualization tools."
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=description)

    p.add_argument("tractograms", metavar="tractogram", nargs="+",
                   help="File(s) containing streamlines (.trk|.tck).")

    # p.add_argument("--ref",
    #                help="Reference frame to display the streamlines in (.nii).")

    # p.add_argument("--colormap", choices=["native", "orientation", "distinct"],
    #                default="native",
    #                help="Choose how the different tractograms should be colored:\n"
    #                     "1) load colors from file, if available\n"
    #                     "2) standard orientation colormap for every streamline\n"
    #                     "3) use distinct color for each tractogram\n"
    #                     "Default: load colors from file if any, otherwise use \n"
    #                     "         a standard orientation colormap for every \n"
    #                     "         streamline")

    return p

class Bundle(object):
    def __init__(self, streamlines, color=None):
        self.streamlines = streamlines
        self.color = color
        self.clusters = None
        self.clusters_colors = []
        self.streamlines_colors = np.ones((len(self.streamlines), 3))

        # Create 3D actor to display this bundle's streamlines.
        self.actor = actor.line(self.streamlines, colors=color)

    def cluster(self, threshold):
        metric = MDF(ResampleFeature(nb_points=20))
        qb = QuickBundles(metric=metric, threshold=threshold)

        self.clusters = qb.cluster(self.streamlines)
        self.clusters_colors = [color for c, color in zip(self.clusters, distinguishable_colormap(bg=(0, 0, 0)))]

        if len(self.clusters) == 1 and self.color is not None:
            # Keep initial color
            self.clusters_colors = [self.color]

        for cluster, color in zip(self.clusters, self.clusters_colors):
            self.streamlines_colors[cluster.indices] = color

        colors = []
        for color, streamline in zip(self.streamlines_colors, self.streamlines):
            colors += [color] * len(streamline)

        vtk_colors = utils.numpy_to_vtk_colors(255 * np.array(colors))
        vtk_colors.SetName("Colors")
        self.actor.GetMapper().GetInput().GetPointData().SetScalars(vtk_colors)

    def get_cluster_as_bundles(self):
        if self.clusters is None:
            raise NameError("Streamlines need to be clustered first!")

        if len(self.clusters) == 1:
            # Keep initial color
            bundle = Bundle(self.streamlines[self.clusters[0].indices], self.color)
            return [bundle]

        bundles = []
        for cluster, color in zip(self.clusters, self.clusters_colors):
            bundle = Bundle(self.streamlines[cluster.indices], color)
            bundles.append(bundle)

        return bundles

class StreamlinesVizu(object):
    def __init__(self, tractogram_filename, savedir="./", screen_size=(1024, 768)):
    # def __init__(self,  tractogram_filename, savedir="./clusters/", screen_size=(1360, 768)):
        self.tractogram_filename = tractogram_filename
        filename, _ = os.path.splitext(os.path.basename(self.tractogram_filename))
        self.savedir = pjoin(savedir, filename)
        self.screen_size = screen_size

        self.tfile = nib.streamlines.load(self.tractogram_filename)
        self.bundles = {}
        self.bundles["/"] = Bundle(self.tfile.streamlines)
        self.root_bundle = "/"
        self.selected_bundle = None
        self.last_threshold = None
        self.other_bundles_visibility_state = "visible"

    def _set_other_bundles_visibility(self, state):
        if state == "visible":
            print("Showing...")
            self.show_dim_hide_button.color = (0, 1, 0)
            self.other_bundles_visibility_state = "visible"
            for k, v in self.bundles.items():
                if k != self.selected_bundle:
                    v.actor.SetVisibility(True)
                    v.actor.GetProperty().SetOpacity(1)

        elif state == "dimmed":
            print("Dimming...")
            self.show_dim_hide_button.color = (0.5, 0.5, 0)
            self.other_bundles_visibility_state = "dimmed"
            for k, v in self.bundles.items():
                if k != self.selected_bundle:
                    v.actor.SetVisibility(True)
                    v.actor.GetProperty().SetOpacity(0.1)

        elif state == "hidden":
            print("Hidding...")
            self.show_dim_hide_button.color = (1, 0, 0)
            self.other_bundles_visibility_state = "hidden"
            for k, v in self.bundles.items():
                if k != self.selected_bundle:
                    v.actor.SetVisibility(False)
                    v.actor.GetProperty().SetOpacity(1)

        else:
            raise ValueError("Unknown visibility state: {}".format(state))


    def _add_bundle_right_click_callback(self, bundle, bundle_name):

        def open_clustering_panel(iren, obj, *args):
            # Set maximum threshold value depending on the selected bundle.
            self.clustering_panel.slider.max_value = bundle.actor.GetLength() / 2.
            self.clustering_panel.slider.set_ratio(0.5)
            self.clustering_panel.slider.update()
            self.clustering_panel.set_visibility(True)
            self.selected_bundle = bundle_name

            # Dim other bundles
            self.bundles[bundle_name].actor.GetProperty().SetOpacity(1)
            self.bundles[bundle_name].actor.SetVisibility(True)
            self._set_other_bundles_visibility("dimmed")
            bundle.cluster(threshold=self.clustering_panel.slider.value)

            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        self.iren.add_callback(bundle.actor, "RightButtonPressEvent", open_clustering_panel)

    def _make_clustering_panel(self):
        # Panel
        size = (self.screen_size[0], self.screen_size[1]//10)
        center = tuple(np.array(size) / 2.)  # Lower left corner of the screen.
        panel = gui_2d.Panel2D(center=center, size=size, color=(1, 1, 1), align="left")

        # "Apply" button
        def animate_button_callback(iren, obj, button):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # button: Button2D
            obj.GetProperty().SetColor(0, 0.5, 0)
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        def apply_button_callback(iren, obj, button):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # button: Button2D
            print("Applying...")

            # Create new actors, one for each new bundle.
            bundles = self.bundles[self.selected_bundle].get_cluster_as_bundles()
            for i, bundle in enumerate(bundles):
                name = "{}{}/".format(self.selected_bundle, i)
                self.bundles[name] = bundle
                self.ren.add(bundle.actor)
                self._add_bundle_right_click_callback(bundle, name)

            # Remove original bundle.
            self.ren.rm(self.bundles[self.selected_bundle].actor)
            del self.bundles[self.selected_bundle]
            self.selected_bundle = None

            # Close panel
            panel.set_visibility(False)

            self._set_other_bundles_visibility("visible")

            # TODO: apply clustering if needed, close panel, add command to history, re-enable bundles context-menu.
            button.color = (0, 1, 0)  # Restore color.
            print("Done.")
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        button = gui_2d.Button2D(icon_fnames={'apply': read_viz_icons(fname='checkmark_neg.png')})
        button.color = (0, 1, 0)
        button.add_callback("LeftButtonPressEvent", animate_button_callback)
        button.add_callback("LeftButtonReleaseEvent", apply_button_callback)
        panel.add_element(button, (0.98, 0.2))

        # "Hide" button
        def toggle_other_bundles_visibility(iren, obj, button):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # button: Button2D

            if self.other_bundles_visibility_state == "visible":
                self._set_other_bundles_visibility("dimmed")

            elif self.other_bundles_visibility_state == "dimmed":
                self._set_other_bundles_visibility("hidden")

            elif self.other_bundles_visibility_state == "hidden":
                self._set_other_bundles_visibility("visible")

            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        self.show_dim_hide_button = gui_2d.Button2D(icon_fnames={'show_dim_hide': read_viz_icons(fname='infinite_neg.png')})
        self.show_dim_hide_button.add_callback("LeftButtonPressEvent", toggle_other_bundles_visibility)
        panel.add_element(self.show_dim_hide_button, (0.02, 0.88))

        # Threshold slider
        def disk_press_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D
            # Only need to grab the focus.
            iren.event.abort()  # Stop propagating the event.

        def disk_move_callback(iren, obj, slider):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # slider: LineSlider2D
            position = iren.event.position
            slider.set_position(position)

            threshold = slider.value
            if self.last_threshold != threshold:
                self.bundles[self.selected_bundle].cluster(threshold)
                self.last_threshold = threshold

            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        slider = gui_2d.LineSlider2D(length=1000, text_template="{value:.1f}mm")
        slider.add_callback("LeftButtonPressEvent", disk_move_callback, slider.slider_line)
        slider.add_callback("LeftButtonPressEvent", disk_press_callback, slider.slider_disk)
        slider.add_callback("MouseMoveEvent", disk_move_callback, slider.slider_disk)
        slider.add_callback("MouseMoveEvent", disk_move_callback, slider.slider_line)
        panel.add_element(slider, (0.5, 0.5))
        panel.slider = slider

        return panel

    def initialize_scene(self):
        self.ren = window.Renderer()
        self.iren = CustomInteractorStyle()
        self.show_m = window.ShowManager(self.ren, size=self.screen_size, interactor_style=self.iren)

        # Add clustering panel to the scene.
        self.clustering_panel = self._make_clustering_panel()
        self.clustering_panel.set_visibility(False)
        self.ren.add(self.clustering_panel)

        # Add "Save" button
        def animate_save_button_callback(iren, obj, button):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # button: Button2D
            obj.GetProperty().SetColor(0.5, 0.5, 0.5)
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        def save_button_callback(iren, obj, button):
            # iren: CustomInteractorStyle
            # obj: vtkActor picked
            # button: Button2D
            print("Saving...")

            for i, k in enumerate(sorted(self.bundles.keys())):
                filename = pjoin(self.savedir, k + ".tck")
                dirname = os.path.dirname(filename)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)

                t = Tractogram(streamlines=self.bundles[k].streamlines,
                               affine_to_rasmm=np.eye(4))
                nib.streamlines.save(t, filename)

            # TODO: apply clustering if needed, close panel, add command to history, re-enable bundles context-menu.
            button.color = (1, 1, 1)  # Restore color.
            print("Done.")
            iren.force_render()
            iren.event.abort()  # Stop propagating the event.

        save_button = gui_2d.Button2D(icon_fnames={'save': read_viz_icons(fname='floppy-disk_neg.png')})
        save_button.color = (1, 1, 1)
        save_button.add_callback("LeftButtonPressEvent", animate_save_button_callback)
        save_button.add_callback("LeftButtonReleaseEvent", save_button_callback)
        save_button.set_center((0.98, 0.98))
        self.ren.add(save_button)

        # Add objects to the scene.
        self.ren.add(self.bundles[self.root_bundle].actor)
        self._add_bundle_right_click_callback(self.bundles[self.root_bundle], self.root_bundle)

        # self.ren.background((1, 0.5, 0))

    def run(self):
        self.show_m.start()


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    vizu = StreamlinesVizu(args.tractograms[0])
    vizu.initialize_scene()
    vizu.run()


if __name__ == "__main__":
    main()
