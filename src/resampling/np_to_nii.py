import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk

def get_np_volume_from_sitk(sitk_image):
    trans = (2, 1, 0)
    pixel_spacing = sitk_image.GetSpacing()
    image_position_patient = sitk_image.GetOrigin()
    np_image = sitk.GetArrayFromImage(sitk_image)
    np_image = np.transpose(np_image, trans)
    return np_image, pixel_spacing, image_position_patient

if __name__ == "__main__":
    
    output_dir = "C:\\Users\\Arthu\\Desktop\\submit_version\\A"
    new_output_dir = os.path.join(output_dir, "nii_output")
    path_to_bbox_file = "C:\\Users\\Arthu\\Documents\\GitHub\\hecktor\\data\\bbox_test.csv"
    ct_dir = "C:\\Users\\Arthu\\Documents\\GitHub\\hecktor\\data\\hecktor_nii"

    if not os.path.exists(new_output_dir):
        os.mkdir(new_output_dir)
    for file_path in glob.glob(os.path.join(output_dir, "*.npy")):
        patient_id = file_path.split('\\')[-1].split('.')[0]
        np_image = np.load(file_path)
        out_path = os.path.join(new_output_dir, "{}.nii.gz".format(patient_id))

        # ct_path = os.path.join(ct_dir, patient_id, "{}_ct.nii.gz".format(patient_id))
        # _, pixel_spacing, image_position_patient = get_np_volume_from_sitk(sitk.ReadImage(ct_path))
        # print(image_position_patient)
        
        trans = (2, 1, 0)
        sitk_image = sitk.GetImageFromArray(np.transpose(np_image, trans))
        bb_df = pd.read_csv(path_to_bbox_file, index_col='PatientID')
        sitk_image.SetSpacing((1.0, 1.0, 1.0))

        origin = bb_df.loc[[patient_id], ['x1','y1','z1']].values[0]

        sitk_image.SetOrigin(origin)
        sitk.WriteImage(sitk_image, out_path)