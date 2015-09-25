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
    #ctf.AddRGBPoint(0.0, 0.230, 0.299, 0.754)
    #ctf.AddRGBPoint(0.5, 0.865, 0.865, 0.865)
    #ctf.AddRGBPoint(1.0, 0.706, 0.016, 0.150)

    ctf.AddRGBPoint(0.0, 0, 0, 1)
    ctf.AddRGBPoint(0.5, 0, 0, 0)
    ctf.AddRGBPoint(1.0, 1, 0, 0)

    #ctf.AddRGBPoint(0.0, 0.78, 0.71, 0.21)
    #ctf.AddRGBPoint(0.5, 0, 0, 0)
    #ctf.AddRGBPoint(1.0, 0.31, 0.87, 0.82)

    # lut = vtk.vtkLookupTable()
    # lut.SetNumberOfTableValues(tableSize)
    # lut.Build()

    # for i in range(0, tableSize):
    #    rgb = list(ctf.GetColor(float(i)/tableSize))+[1]
    #    lut.SetTableValue(i, rgb)

    colormap = [[255, 125, 0], [254, 123, 0], [252, 121, 0], [250, 119, 0], [248, 117, 0], [246, 115, 0], [243, 113, 0], [241, 111, 0], [239, 109, 0], [237, 108, 0], [235, 106, 0], [233, 104, 0], [231, 102, 0], [229, 100, 0], [227, 98, 0], [225, 96, 0], [223, 94, 0], [221, 92, 0], [219, 90, 0], [217, 88, 0], [215, 86, 0], [213, 84, 0], [211, 82, 0], [209, 80, 0], [207, 78, 0], [205, 76, 0], [203, 74, 0], [200, 72, 0], [198, 70, 0], [196, 68, 0], [194, 66, 0], [192, 64, 0], [190, 62, 0], [188, 60, 0], [186, 58, 0], [184, 56, 0], [182, 54, 0], [180, 53, 0], [178, 51, 0], [176, 49, 0], [174, 47, 0], [172, 45, 0], [170, 43, 0], [168, 41, 0], [166, 39, 0], [164, 37, 0], [162, 35, 0], [159, 33, 0], [157, 31, 0], [155, 29, 0], [153, 27, 0], [151, 25, 0], [149, 23, 0], [147, 21, 0], [145, 19, 0], [143, 17, 0], [141, 15, 0], [139, 13, 0], [137, 11, 0], [135, 9, 0], [133, 7, 0], [131, 5, 0], [129, 3, 0], [127, 1, 0], [125, 0, 0], [123, 0, 0], [121, 0, 0], [119, 0, 0], [117, 0, 0], [115, 0, 0], [113, 0, 0], [111, 0, 0], [109, 0, 0], [107, 0, 0], [105, 0, 0], [103, 0, 0], [101, 0, 0], [99, 0, 0], [97, 0, 0], [95, 0, 0], [93, 0, 0], [91, 0, 0], [89, 0, 0], [87, 0, 0], [85, 0, 0], [83, 0, 0], [81, 0, 0], [80, 0, 0], [78, 0, 0], [76, 0, 0], [74, 0, 0], [72, 0, 0], [70, 0, 0], [68, 0, 0], [66, 0, 0], [64, 0, 0], [62, 0, 0], [60, 0, 0], [58, 0, 0], [56, 0, 0], [54, 0, 0], [52, 0, 0], [50, 0, 0], [48, 0, 0], [46, 0, 0], [44, 0, 0], [42, 0, 0], [40, 0, 0], [38, 0, 0], [36, 0, 0], [34, 0, 0], [32, 0, 0], [30, 0, 0], [28, 0, 0], [27, 0, 0], [25, 0, 0], [23, 0, 0], [21, 0, 0], [19, 0, 0], [17, 0, 0], [15, 0, 0], [13, 0, 0], [11, 0, 0], [9, 0, 0], [7, 0, 0], [5, 0, 0], [3, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 3], [0, 0, 5], [0, 0, 7], [0, 0, 9], [0, 0, 11], [0, 0, 13], [0, 0, 15], [0, 0, 17], [0, 0, 19], [0, 0, 21], [0, 0, 23], [0, 0, 25], [0, 0, 27], [0, 0, 29], [0, 0, 31], [0, 0, 33], [0, 0, 35], [0, 0, 37], [0, 0, 39], [0, 0, 41], [0, 0, 44], [0, 0, 46], [0, 0, 48], [0, 0, 50], [0, 0, 52], [0, 0, 54], [0, 0, 56], [0, 0, 58], [0, 0, 60], [0, 0, 62], [0, 0, 64], [0, 0, 66], [0, 0, 68], [0, 0, 70], [0, 0, 72], [0, 0, 74], [0, 0, 76], [0, 0, 78], [0, 0, 80], [0, 0, 82], [0, 0, 84], [0, 0, 86], [0, 0, 88], [0, 0, 90], [0, 0, 92], [0, 0, 94], [0, 0, 96], [0, 0, 98], [0, 0, 100], [0, 0, 102], [0, 0, 104], [0, 0, 106], [0, 0, 108], [0, 0, 110], [0, 0, 112], [0, 0, 114], [0, 0, 116], [0, 0, 118], [0, 0, 120], [0, 0, 122], [0, 0, 124], [0, 0, 127], [0, 0, 129], [0, 1, 131], [0, 3, 133], [0, 5, 135], [0, 7, 136], [0, 9, 138], [0, 11, 140], [0, 13, 142], [0, 15, 144], [0, 17, 146], [0, 19, 148], [0, 21, 150], [0, 23, 152], [0, 25, 154], [0, 27, 156], [0, 29, 158], [0, 31, 160], [0, 33, 162], [0, 35, 164], [0, 37, 166], [0, 39, 168], [0, 41, 170], [0, 43, 172], [0, 45, 174], [0, 47, 176], [0, 49, 178], [0, 51, 180], [0, 53, 182], [0, 55, 184], [0, 57, 186], [0, 59, 188], [0, 61, 190], [0, 63, 192], [0, 65, 194], [0, 67, 196], [0, 69, 198], [0, 71, 200], [0, 73, 202], [0, 75, 204], [0, 77, 206], [0, 79, 208], [0, 81, 210], [0, 82, 212], [0, 84, 214], [0, 86, 216], [0, 88, 218], [0, 90, 220], [0, 92, 222], [0, 94, 224], [0, 96, 226], [0, 98, 228], [0, 100, 230], [0, 102, 232], [0, 104, 234], [0, 106, 236], [0, 108, 238], [0, 110, 240], [0, 112, 242], [0, 114, 244], [0, 116, 246], [0, 118, 248], [0, 120, 250], [0, 122, 252], [0, 124, 254], [0, 126, 255]]
    print len(colormap)

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(len(colormap))
    lut.Build()

    for i, rgb in enumerate(colormap[::-1]):
        lut.SetTableValue(i, tuple(np.array(rgb)/255.) + (1,))

    return lut


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    if args.tractograms is not None:
        parser.error("Comparing streamlines is not implemented yet.")

    # Create a jet lookup table.
    lut = MakeLUTFromCTF(40)

    actors = []
    value_ranges = []
    for anat_filename in args.anats:
        print anat_filename
        anat = nib.load(anat_filename)
        data = anat.get_data()

        # Zero should be in the middle of the range.
        max_abs_value = max(abs(data.min()), abs(data.max()))
        value_range = (-max_abs_value, max_abs_value)
        print value_range

        lut = MakeLUTFromCTF(256)
        lut.SetTableRange(0, 255)

        slice_actor = actor.slicer(data, np.eye(4), value_range, lookup_colormap=lut)
        #slice_actor = actor.slicer(data, anat.get_affine(), value_range, lookup_colormap=lut)
        slice_actor.display(None, None, slice_actor.shape[2]//2)
        slice_actor.SetInterpolate(False)

        value_ranges.append(value_range)
        actors.append(slice_actor)

    grid_layout = layout.GridLayout(cell_padding=10)
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
