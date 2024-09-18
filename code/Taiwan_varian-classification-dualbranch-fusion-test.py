import logging
import os
import sys
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import monai
from monai.data import CSVSaver
from monai.transforms import AddChanneld, Compose, LoadImaged, Resized, ScaleIntensityd, EnsureTyped
from networks.densenet_dualbranch_fusion import DenseNet

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)

    # gather all data 

    # FILL HERE
    data_path_t1 = r'E:\Yuli\Projects\overian_classification\overian_classification\dataset\JHH_Penn_Taiwan_dataset\Taiwan_nii\T1-ROI-preprocess' #FILL HERE
    data_path_t2 = r'E:\Yuli\Projects\overian_classification\overian_classification\dataset\JHH_Penn_Taiwan_dataset\Taiwan_nii\T2-ROI-preprocess' #FILL HERE
    xx = pd.read_excel(r'E:\Yuli\Projects\overian_classification\overian_classification\dataset\JHH_Penn_Taiwan_dataset\Taiwan_nii\label-classifyT1C+T2-clear.xlsx', sheet_name='Sheet1') #FILL HERE
    # END FILL

    images_all = xx.iloc[:, 0].tolist()  # all T1C images
    lab = xx.iloc[:, 2].tolist()  # all labels
    labels_all = np.array(lab, dtype=np.int64)
    images = images_all[:]  # test images
    labels = labels_all[:]  # test labels

    clinic = xx.iloc[:, 2:9]  # all clinic data
    age = np.array(clinic['age'].tolist(), dtype=float)
    ca125 = np.array(clinic['ca125'].tolist(), dtype=float)
    isca125 = np.array(clinic['isca125'].tolist(), dtype=float)
    menopause = np.array(clinic['menopause'].tolist(), dtype=float)
    ismenopause = np.array(clinic['ismenopause'].tolist(), dtype=float)
    scm = np.array(clinic['scm'].tolist(), dtype=float)
    isscm = np.array(clinic['isscm'].tolist(), dtype=float)

    clinic1 = np.zeros(shape=[29, 7])
    clinic1[:, 0] = age
    clinic1[:, 1] = ca125
    clinic1[:, 2] = isca125
    clinic1[:, 3] = menopause
    clinic1[:, 4] = ismenopause
    clinic1[:, 5] = scm
    clinic1[:, 6] = isscm

    clinic_data = clinic1
    clinics = clinic_data[:, :]  # stores all clinic data

    images_t1 = [os.sep.join([data_path_t1, f]) for f in images] #access t1 preprocessed image path
    print(images_t1)
    images_t2 = [os.sep.join([data_path_t2, f]) for f in images] #access t2 preprocessed image path
    
    #Iterate over tuple and create dictionary in val_files so that each dictionary entry has set of files
    val_files = [
        {
            "img_t1": img_t1,# + '.nii.gz',
            "img_t2": img_t2,# + '.nii.gz',
            "label": label,
            "cli": cli
        } 
        for img_t1, img_t2, label,cli in zip(images_t1[:], images_t2[:], labels[:],clinics[:,:])
    ]  # test images,labels
    print(val_files)

    #
    val_transforms=Compose([LoadImaged(keys=["img_t1","img_t2"]),
                            AddChanneld(keys=["img_t1","img_t2"]),
                            ScaleIntensityd(keys=["img_t1","img_t2"]),
                            Resized(keys=["img_t1","img_t2"],spatial_size=(96,96,96)),
                            EnsureTyped(keys=["img_t1","img_t2"])])

    #
    val_ds=monai.data.Dataset(data=val_files,transform=val_transforms)
    val_loader=DataLoader(val_ds,batch_size=1,num_workers=4,pin_memory=torch.cuda.is_available())

    #
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model=monai.networks.nets.DenseNet121(spatial_dims=3,in_channels=1,out_channels=2).to(device)
    model = DenseNet(spatial_dims=3, in_channels=1, out_channels=2, init_features=64,
                      growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, act=('relu', {'inplace': True}),
                      norm='batch', dropout_prob=0.0).to(device)

    #FILL HERE 
    model.load_state_dict(torch.load("E:\Yuli\Projects\overian_classification\overian_classification\T1-2-classification-model-with-clinical-dymlp-dual\model_classification3d_dict850.pth"))
    #END FILL
    model.eval()
    with torch.no_grad():
        num_correct=0
        metric_count=0
        saver=CSVSaver(output_dir=".\output-T2-dyMLP") #FILL HERE
        for val_data in val_loader:
            val_images_t1,val_images_t2,val_labels,val_cli=val_data["img_t1"].to(device),val_data["img_t2"].to(device),val_data["label"].to(device),val_data["cli"].to(device)
            # val_outputs=model(val_images,val_cli).argmax(dim=1)
            feature, val_outputs = model(val_images_t1,val_images_t2,val_cli)
            val_outputs1 = val_outputs.argmax(dim=1)
            val_outputs2 = F.softmax(val_outputs, dim=1)  # final column is the probability
            value=torch.eq(val_outputs1,val_labels)
            print(val_outputs1, val_labels, value)
            metric_count+=len(value)
            num_correct+=value.sum().item()
            saver.save_batch(val_outputs2)
        metric=num_correct/(metric_count)
        print(num_correct,"/",metric_count)
        print("evalution metric:",metric)
        saver.finalize()

if __name__=="__main__":
    main()


