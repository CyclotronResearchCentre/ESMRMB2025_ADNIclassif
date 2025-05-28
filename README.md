# ESMRMB2025_ADNIclassif



## Overview
Study ID: ESMRMB 2025 #45945
Title: Radiomic Feature Robustness Affected by Magnetic Field Strength in Alzheimer's Disease
This repository reproduces findings from our study analyzing the impact of different MRI field strengths (1.5T vs. 3.0T) on radiomic feature robustness in Alzheimer's Disease (AD). The pipeline extracts quantitative features from MRI scans and evaluates their cross-field-strength consistency for AD diagnosis.

## Data Preparation
Download required data from the ADNI Database:
https://adni.loni.usc.edu
Cohorts: AD patients vs. Healthy Controls

## Pipeline
**Step1: Hippocampus Segmentation****

Bilateral hippocampus segmentation using HippoDeep

**Step 2: Radiomic Feature Extraction**
Extracts 106 radiomic features using PyRadiomics

Features include:
First-order statistics
3D shape features
Texture features (GLCM, GLDM, GLSZM)

**step3: Robustness Analysis
Nonlinear Concordance Correlation Coefficient (NCCC)

Measures feature agreement between 1.5T/3.0T paired scans

Threshold: NCCC > 0.6 indicates robustness

Wilcoxon Signed-Rank Test (WSR)

Identifies features with not significant field-strength-dependent variations (p>0.05)

**step4: Radiomic
Feature selection:
Mutual information scores


Classifiers:

Random Forest (350 estimators)


Validation:

5-fold cross-validation


Metrics:

AUC

## Reference
[1] Das, Oindrila. ”Progression of Alzheimer’s Disease: A Neuropsychological Perspective.” Available at SSRN 5118011 (2025). 
[2] Bevilacqua, Roberta, et al. ”Radiomics and artificial intelligence for the diagnosis and monitoring of Alzheimer’s disease: a systematic review of studies in the field.” Journal of clinical medicine 12.16 (2023): 5432 
[3] Jytzler, J. A., and Simon Lysdahlgaard. ”Radiomics evaluation for the early detection of Alzheimer’s dementia using T1-weighted MRI.” Radiography 30.5 (2024): 1427-1433. 
[4] Zhao, Kun, et al. ”A neuroimaging biomarker for Individual Brain-Related Abnormalities In Neurodegeneration (IBRAIN): a cross-sectional study.” EClinicalMedicine 65 (2023). 
[5] Thyreau, Benjamin, et al. ”Segmentation of the hippocampus by transferring algorithmic knowledge for large cohort processing.” Medical image analysis 43 (2018): 214-228. 
[6] Van Griethuysen, Joost JM, et al. ”Computational radiomics system to decode the radiographic phenotype.” Cancer research 77.21 (2017): e104-e107.