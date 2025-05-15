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
from HD_BET import hd_bet_prediction
import SimpleITK as sitk
from pathlib import Path
import torch
import glob
from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from HD_BET.checkpoint_download import maybe_download_parameters
from concurrent.futures import ThreadPoolExecutor


# Set paths to data folders and CSV metadata file
folder_1_5T = "/mnt/data/FRA/dataset/1.5T/0-3_months"
folder_3T = "/mnt/data/FRA/dataset/3T/0-3_months"
# folder_1_5T = "/mnt/data/FRA/dataset/1.5T/test"
# folder_3T = "/mnt/data/FRA/dataset/3T/test"
csv_file = "info.csv"  
output_dir="./result"




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


def brain_extraction(nifti_path, output_dir, device="cpu"):
    """Skull stripping using HD-BET"""
    brain_path=str(os.path.join(output_dir, "brain.nii.gz"))
    maybe_download_parameters()

    predictor = get_hdbet_predictor(
        use_tta=False,
        device=torch.device('cpu'),
        verbose=False
    )
    # Run HD-BET
    hdbet_predict(
    input_file_or_folder=str(nifti_path),
    output_file_or_folder=brain_path,
    predictor=predictor,  
    keep_brain_mask=True,          
    compute_brain_extracted_image=True )
    
    # Load and return stripped brain
    return sitk.ReadImage(brain_path)

def hippo_segment(nifit_dir):
    input_path = Path(nifit_dir).absolute()
    cmd = ["uv", "run", "/mnt/data/FRA/hippodeep_pytorch-master/hippodeep.py", str(input_path)]
    result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=input_path.parent    
    )   
def wrapper(subject_dir, mode, folder, df_info):
    subject_path = glob.glob(os.path.join(folder, subject_dir, '*', '*'))[0]
    process_subject(subject_dir, mode, subject_path, df_info)

def process_subject(subject_id, t_type, dicom_dir, df_info, device="cpu"):
    output_path = Path(output_dir) / subject_id / t_type
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        # 1. Convert DICOM to NIfTI
        nifti_path = output_path / "original.nii.gz"
        converted_path, dicom_metadata = convert_dicom_to_nifti(dicom_dir, str(nifti_path))
        
        # 2. Skull stripping with HD-BET
        brain_image = brain_extraction(nifti_path , output_path, device="cpu")
        
        # 3. Hippocampus segmentation (Hippo-deeo)
        # Convert to numpy array with proper orientation
        hippo_segment(nifti_path)
    except Exception as e:
        print(f"Failed processing {subject_id}: {str(e)}")
        return None
def main():
    #create output dir
    Path(output_dir).mkdir(exist_ok=True)
     
    df_info = pd.read_csv(csv_file)
    
    feature_1_5T =[]
    feature_3T =[]
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     # 1.5T
    #     futures_15T = [
    #         executor.submit(wrapper, subject_dir, "1.5T", folder_1_5T, df_info)
    #         for subject_dir in os.listdir(folder_1_5T)
    #     ]

    #     # 3T
    #     futures_3T = [
    #         executor.submit(wrapper, subject_dir, "3T", folder_3T, df_info)
    #         for subject_dir in os.listdir(folder_3T)
    #     ]

       
    #     for f in futures_15T + futures_3T:
    #         f.result()
    
    # for subject_dir in os.listdir(folder_1_5T):
    #     subject_path = glob.glob(os.path.join(folder_1_5T, subject_dir,'*','*'))[0]
    #     process_subject(subject_dir, "1.5T", subject_path, df_info)
    #     # if os.path.isdir(subject_path):
    #     #     features = process_subject(subject_dir, "1.5T", subject_path, df_info)
    #     #     if features: feature_1_5T.append(features)

    # for subject_dir in os.listdir(folder_3T):
    #     subject_path = glob.glob(os.path.join(folder_3T, subject_dir,'*','*'))[0] 
    #     process_subject(subject_dir, "3T", subject_path, df_info)
        # if os.path.isdir(subject_path):
        #     features = process_subject(subject_dir, "3T", subject_path, df_info)
        #     if features: Feature_3T.append(features)
    # df_features_1_5T = pd.DataFrame(feature_1_5T)
    # df_features_1_5T.to_csv(os.path.join(output_dir, "1.5T_features.csv"), index=False)
    # df_features_3T = pd.DataFrame(feature_3T)
    # df_features_3T.to_csv(os.path.join(output_dir, "3T_features.csv"), index=False)

if __name__ == "__main__":
    main()
