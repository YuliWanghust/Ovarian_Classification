# Ovarian Classification

Code for paper [Deep Learning Models to Classify Benign and Malignant Ovarian Lesions on MRI: A Multicenter Study]()

<img src="Pics/workflow.png" align="middle" width="75%">

Our study aims to develop a comprehensive end-to-end pipeline. We first evaluate SAM's performance in segmenting ovarian lesions on MRI. Building on this, we then develop a robust DL model for distinguishing between benign and malignant ovarian lesions using MRI data. By leveraging SAM-segmented images, we seek to enhance the effectiveness and generalizability of our diagnostic approach, ultimately improving clinical outcomes for patients. The workflow depicting the integration of T1CE and T2WI with ROI segmentation, pre-processed along with clinical data, into the deep learning model for lesion classification.

## Note:
```./code``` and ```./dataset``` contain the code for each component of Ovarian classification model and pre-process code for the both T1 and T2 dataset. These programs cannot be run with the full T1/T2 ovarian dataset, but help illustrate the exact steps we did to form our cohort.

```./3.	T1-2 classification model-with-clinical-dymlp-dual'' contains the model weights for the ovarian classification model. See the respective README.md in the subfolder to see how to run each component.

## Dependencies
To establish the environment, run this code in the shell:
```
conda env create -f vote_MI.yml
conda activate vote_MI
pip install -e .
```
That will create the environment selective_annotation we used.
