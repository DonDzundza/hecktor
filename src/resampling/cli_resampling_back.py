import os
from multiprocessing import Pool
import glob

import click
import logging
import pandas as pd

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import affine_transform
from scipy.interpolate import RegularGridInterpolator




@click.command()
@click.argument('input_folder',
                type=click.Path(exists=True),
                default='data/segmentation_output_renamed/A')
@click.argument('output_folder',
                type=click.Path(),
                default='data/segmentation_output_tosubmit')
@click.argument('bounding_boxes_file',
                type=click.Path(),
                default='data/bbox_test.csv')
@click.argument('original_resolution_file',
                type=click.Path(),
                default='data/original_resolution_test.csv')
@click.option('--cores',
              type=click.INT,
              default=1,
              help='The number of workers for parallelization.')
@click.option('--order',
              type=click.INT,
              nargs=1,
              default=3,
              help='The order of the spline interpolation used to resample')
def main(input_folder, output_folder, bounding_boxes_file,
         original_resolution_file, cores, order):
    """ This command line interface allows to resample NIFTI files back to the
        original resolution contained in ORIGINAL_RESOLUTION_FILE (this file
        can be gerenated with the file src/resampling/cli_get_resolution.py).
        It also needs the bounding boxes contained in BOUNDING_BOXES_FILE.
        The images are resampled with spline interpolation
        of degree --order (default=3) and the segmentation are resampled by
        nearest neighbor interpolation.

        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    bb_df = pd.read_csv(bounding_boxes_file)
    bb_df = bb_df.set_index('PatientID')
    resolution_df = pd.read_csv(original_resolution_file)
    resolution_df = resolution_df.set_index('PatientID')
    files_list = [
        f for f in glob.glob(input_folder + '/*.nii.gz', recursive=True)
    ]
    print(files_list)
    resampler = Resampler(bb_df, output_folder, order)
    patient_list = [f.split('\\')[-1].split('.')[0] for f in files_list]
    resolution_list = [(resolution_df.loc[k, 'Resolution_x'],
                        resolution_df.loc[k, 'Resolution_y'],
                        resolution_df.loc[k, 'Resolution_z'])
                       for k in patient_list]
    with Pool(cores) as p:
        p.starmap(resampler, zip(files_list, resolution_list))
    # assert len(files_list) == len(resolution_list)
    # for i in range(0, len(files_list)):
    #     resampler(files_list[i], resolution_list[i])

class Resampler():
    def __init__(self,
                 bb_df,
                 output_folder,
                 order,
                 resampling=None,
                 logger=None):
        super().__init__()
        self.bb_df = bb_df
        self.output_folder = output_folder
        self.resampling = resampling
        self.order = order
        self.logger = logger

    def __call__(self, f, resampling=None):
        if resampling is None:
            resampling = self.resampling
        patient_name = f.split('\\')[1].split('.')[0]
        # patient_folder = os.path.join(self.output_folder, patient_name)
        # if not os.path.exists(patient_folder):
        #     os.mkdir(patient_folder)
        # output_file = os.path.join(patient_folder, f.split('/')[-1])
        output_file = os.path.join(self.output_folder, f.split('\\')[-1])
        bb = (self.bb_df.loc[patient_name, 'x1'], self.bb_df.loc[patient_name,
                                                                 'y1'],
              self.bb_df.loc[patient_name, 'z1'], self.bb_df.loc[patient_name,
                                                                 'x2'],
              self.bb_df.loc[patient_name, 'y2'], self.bb_df.loc[patient_name,
                                                                 'z2'])
        print('Resampling patient {}'.format(patient_name))

        resample_and_crop(f,
                          output_file,
                          bb,
                          resampling=resampling,
                          order=self.order)


def resample_and_crop(input_file,
                      output_file,
                      bounding_box,
                      resampling=(1.0, 1.0, 1.0),
                      order=3):
    np_volume, pixel_spacing, origin = get_np_volume_from_sitk(
        sitk.ReadImage(input_file))
    resampling = np.asarray(resampling)
    # If one value of resampling is -1 replace it with the original value
    for i in range(len(resampling)):
        if resampling[i] == -1:
            resampling[i] = pixel_spacing[i]
        elif resampling[i] < 0:
            raise ValueError(
                'Resampling value cannot be negative, except for -1')

    np_volume = resample_np_binary_volume(np_volume, origin, pixel_spacing,
                                            resampling, bounding_box)


    origin = np.asarray([bounding_box[0], bounding_box[1], bounding_box[2]])
    sitk_volume = get_sitk_volume_from_np(np_volume, resampling, origin)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_file)
    writer.SetImageIO("NiftiImageIO")
    writer.Execute(sitk_volume)


def resample_np_volume(np_volume,
                       origin,
                       current_pixel_spacing,
                       resampling_px_spacing,
                       bounding_box,
                       order=3):

    zooming_matrix = np.identity(3)
    zooming_matrix[0, 0] = resampling_px_spacing[0] / current_pixel_spacing[0]
    zooming_matrix[1, 1] = resampling_px_spacing[1] / current_pixel_spacing[1]
    zooming_matrix[2, 2] = resampling_px_spacing[2] / current_pixel_spacing[2]

    offset = ((bounding_box[0] - origin[0]) / current_pixel_spacing[0],
              (bounding_box[1] - origin[1]) / current_pixel_spacing[1],
              (bounding_box[2] - origin[2]) / current_pixel_spacing[2])

    output_shape = np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing

    np_volume = affine_transform(np_volume,
                                 zooming_matrix,
                                 offset=offset,
                                 mode='mirror',
                                 order=order,
                                 output_shape=output_shape.astype(int))

    return np_volume


def grid_from_spacing(start, spacing, n):
    return np.asarray([start + k * spacing for k in range(n)])


def resample_np_binary_volume(np_volume, origin, current_pixel_spacing,
                              resampling_px_spacing, bounding_box):

    x_old = grid_from_spacing(origin[0], current_pixel_spacing[0],
                              np_volume.shape[0])
    y_old = grid_from_spacing(origin[1], current_pixel_spacing[1],
                              np_volume.shape[1])
    z_old = grid_from_spacing(origin[2], current_pixel_spacing[2],
                              np_volume.shape[2])

    output_shape = (np.ceil([
        bounding_box[3] - bounding_box[0],
        bounding_box[4] - bounding_box[1],
        bounding_box[5] - bounding_box[2],
    ]) / resampling_px_spacing).astype(int)

    x_new = grid_from_spacing(bounding_box[0], resampling_px_spacing[0],
                              output_shape[0])
    y_new = grid_from_spacing(bounding_box[1], resampling_px_spacing[1],
                              output_shape[1])
    z_new = grid_from_spacing(bounding_box[2], resampling_px_spacing[2],
                              output_shape[2])
    interpolator = RegularGridInterpolator((x_old, y_old, z_old),
                                           np_volume,
                                           method='nearest',
                                           bounds_error=False,
                                           fill_value=0)
    x, y, z = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    pts = np.array(list(zip(x.flatten(), y.flatten(), z.flatten())))

    return interpolator(pts).reshape(output_shape)

def get_sitk_volume_from_np(np_image, pixel_spacing, image_position_patient):
    trans = (2, 1, 0)
    sitk_image = sitk.GetImageFromArray(np.transpose(np_image, trans))
    sitk_image.SetSpacing(pixel_spacing)
    sitk_image.SetOrigin(image_position_patient)
    return sitk_image


def get_np_volume_from_sitk(sitk_image):
    trans = (2, 1, 0)
    pixel_spacing = sitk_image.GetSpacing()
    image_position_patient = sitk_image.GetOrigin()
    np_image = sitk.GetArrayFromImage(sitk_image)
    np_image = np.transpose(np_image, trans)
    return np_image, pixel_spacing, image_position_patient

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
