#! /usr/bin/env python

import vtk
import numpy as np
import nibabel as nib

from dipy.fixes import argparse
from dipy.viz import window, actor, layout


def build_args_parser():
    description = "Compare multiple anatomies/streamlines."
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=description)

    p.add_argument("--anats", metavar="anat", nargs="+",
                   help="File(s) containing streamlines (.nii|.nii.gz).")

    p.add_argument("--tractograms", metavar="tractogram", nargs="+",
                   help="File(s) containing streamlines (.trk).")

    p.add_argument("--ref",
                   help="Reference frame to display the streamlines in (.nii|.nii.gz).")

    p.add_argument("--colormap", choices=["native", "orientation", "distinct"],
                   default="native",
                   help="Choose how the different tractograms should be colored:\n"
                        "1) load colors from file\n"
                        "2) standard orientation colormap for every streamline\n"
                        "3) use distinct color for each tractogram\n"
                        "Default: load colors from file if any, otherwise use \n"
                        "         a standard orientation colormap for every \n"
                        "         streamline")

    return p


def MakeLUTFromCTF(tableSize):
    '''
    Use a color transfer Function to generate the colors in the lookup table.
    See: http://www.vtk.org/doc/nightly/html/classvtkColorTransferFunction.html
    :param: tableSize - The table size
    :return: The lookup table.
    '''
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    # Blue to red.
    ctf.AddRGBPoint(0.0, 0.230, 0.299, 0.754)
    ctf.AddRGBPoint(0.5, 0.865, 0.865, 0.865)
    ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150)

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(tableSize)
    lut.Build()

    for i in range(0, tableSize):
        rgb = list(ctf.GetColor(float(i)/tableSize))+[1]
        lut.SetTableValue(i, rgb)

    return lut


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    if args.tractograms is not None:
        parser.error("Comparing streamlines is not implemented yet.")

    # Create a jet lookup table.
    lut = MakeLUTFromCTF(501)

    actors = []
    value_ranges = []
    for anat_filename in args.anats:
        anat = nib.load(anat_filename)
        data = anat.get_data()

        value_range = (data.min(), data.max())
        lut.SetRange(value_range)

        slice_actor = actor.slicer(data, anat.get_affine(), value_range, lookup_colormap=lut)
        slice_actor.display(None, None, slice_actor.shape[2]//2)

        value_ranges.append(value_range)
        actors.append(slice_actor)

    grid_layout = layout.GridLayout(padding=10)
    grid = actor.Container(layout=grid_layout)
    grid.add(*actors)

    ren = window.Renderer()
    show_m = window.ShowManager(ren, size=(800, 600), interactor_style="image")

    def change_slice(slice_actor, direction, modifier=False):
        z = slice_actor.GetDisplayExtent()[5]
        offset = np.sign(direction)
        if modifier:
            offset *= 10

        slice_actor.display(None, None, z+offset)
        new_z = slice_actor.GetDisplayExtent()[5]
        slice_actor.AddPosition(0, 0, z-new_z)

    def change_slice_event(obj, direction):
        event_position = obj.GetEventPosition()
        picker = obj.GetPicker()

        ren = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
        picked = picker.Pick(event_position[0], event_position[1], 0, ren)

        if picked:
            obj_picked = picker.GetProp3D()

            if isinstance(obj_picked, vtk.vtkImageActor):
                change_slice(obj_picked, direction, obj.GetControlKey())
                obj.Render()

        else:
            for a in actors:
                change_slice(a, direction, obj.GetControlKey())

            obj.Render()

    def change_slice_up(obj, event):
        change_slice_event(obj, -1)

    def change_slice_down(obj, event):
        change_slice_event(obj, 1)

    show_m.iren.RemoveObservers("MouseWheelForwardEvent")
    show_m.iren.AddObserver("MouseWheelForwardEvent", change_slice_up)
    show_m.iren.RemoveObservers("MouseWheelBackwardEvent")
    show_m.iren.AddObserver("MouseWheelBackwardEvent", change_slice_down)

    bg = (1, 1, 1)
    ren.background(bg)
    ren.projection("parallel")

    ren.add(grid)

    ren.reset_camera_tight()
    show_m.start()


if __name__ == "__main__":
    main()
