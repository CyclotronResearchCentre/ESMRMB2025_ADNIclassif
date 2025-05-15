"""
extract the features from whole brain of 1.5T and 3T 



Author: Jiqing Huang
Date: 17/04/2025
Description: 
 

Input Files:
- 1.5 T1 image
- 3T T1 image
- label: NC MCI AND DEMENTIA or AD
Output Files:
features of patients
"""
import os
import subprocess
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import glob
import numpy as np
import nibabel as nib
from radiomics import featureextractor

# Set paths to data folders and CSV metadata file

csv_file = "info.csv"  
input_dir="./result"
output_dir="./features"



# DICOM-specific processing functions
def load_dicom_series(dicom_dir):
    """Load DICOM series with proper sorting and metadata handling"""
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    
    # Handle DICOM metadata
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    
    try:
        image = reader.Execute()
    except RuntimeError as e:
        print(f"DICOM read error in {dicom_dir}: {str(e)}")
        raise
    
    # Store DICOM metadata for later reference
    metadata = {
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection()
    }
    
    return image, metadata

def convert_dicom_to_nifti(dicom_dir, output_path):
    """Convert DICOM series to NIfTI for compatibility with HD-BET"""
    image, metadata = load_dicom_series(dicom_dir)
    sitk.WriteImage(image, output_path)
    return output_path, metadata


def hippo_image_extract(subject_path):
    original_img = nib.load( subject_path/ "original.nii.gz")
    hippo_L = nib.load(subject_path/ "original_mask_L.nii.gz").get_fdata()
    hippo_R = nib.load(subject_path/ "original_mask_R.nii.gz").get_fdata()
    hippo_mask = (hippo_L > 0)|(hippo_R > 0)
    hippo_image = original_img.get_fdata() * hippo_mask
    hippo_data = nib.Nifti1Image(hippo_image, affine=original_img.affine, header= original_img.header)
    hippo_datamask= nib.Nifti1Image(hippo_mask, affine=original_img.affine, header= original_img.header)
    nib.save(hippo_data, subject_path / "hippo.nii.gz")
    nib.save(hippo_datamask, subject_path / "hippo_mask.nii.gz")
    sitk_image = sitk.GetImageFromArray(hippo_image.astype(np.float32).transpose(2, 1, 0))
    sitk_mask = sitk.GetImageFromArray(hippo_mask.astype(np.uint8).transpose(2, 1, 0))
    reference_image = sitk.ReadImage(str(subject_path / "original.nii.gz"))
    sitk_image.CopyInformation(reference_image)
    sitk_mask.CopyInformation(reference_image)

    return sitk_image, sitk_mask


def save_features_to_csv(features_dict, subject_id, research_group, output_file,scan_type):
    # Convert the feature dictionary to a DataFrame
    features_df = pd.DataFrame(list(features_dict.items()), columns=["Feature", "Value"])

    # Add subject metadata
    features_df["subject_id"] = subject_id
    features_df["research_group"] = research_group
    features_df["scan_type"] = scan_type  # Indicate the scan type (e.g., 1.5T or 3T)

    # Save to CSV: append mode, write header only if file doesn't exist
    features_df.to_csv(output_file, mode='a', header=not output_file.exists(), index=False)



def process_subject(subject_id, input_dir,research_group):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        # 1. calculate hippo image
        path_1_5T =  input_dir / "1.5T"
        hippo_1_5T,mask_1_5T = hippo_image_extract(path_1_5T)
        path_3T =  input_dir / "3T"
        hippo_3T,mask_3T =hippo_image_extract(path_3T)
        print(f"[✔] Processed subject: {subject_id}")
    except Exception as e:
        print(f"[✘] Failed to process subject {subject_id}: {e}")
def main():
    #create output dir
    Path(output_dir).mkdir(exist_ok=True)
     
    df_info = pd.read_csv(csv_file)
    
    feature_1_5T =[]
    feature_3T =[]
    for subject_dir in os.listdir(input_dir):
        subject_path = Path(glob.glob(os.path.join(input_dir, subject_dir))[0])
        research_group = df_info.loc[df_info['subject_id'] == subject_dir, 'research_group'].values[0]
        process_subject(subject_dir,subject_path,research_group)
   
if __name__ == "__main__":
    main()
