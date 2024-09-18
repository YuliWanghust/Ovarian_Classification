# spacing    in training & testing
# crop
import nibabel as nb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#FILL HERE
data = pd.read_excel('./label-T1C+T2-clear.xlsx',sheet_name='Sheet1')
#END FILL

for i in range(1):
   print(i)
   #roi=np.zeros([481,363,113])
   label = nb.load(data.iloc[i, 5]) #T1 mask-col3 T1 image-col2;  T2 mask-col5 T2 image-col4
   image = nb.load(data.iloc[i, 4])
   label_data = label.get_fdata()
   image_data=image.get_fdata()
   affine=label.affine.copy()
   hdr=label.header.copy()

   x,y,z=np.nonzero(label_data)

   roi=image_data[np.min(x):np.max(x)+1,np.min(y):np.max(y)+1,np.min(z):np.max(z)+1] #roi


   # plt.imshow(roi[:,:,0])
   # plt.show()
   # plt.imshow(roi[:, :,56])
   # plt.show()

   #FILL HERE
   file='./T2-ROI-preprocess/'+str(data.iloc[i,0])+'.nii.gz'
   #END FILL

   new_roi=nb.Nifti1Image(roi,affine,hdr)
   nb.save(new_roi,file)
   # plt.imshow(roi[:,:,0])
   # plt.show()
   del x,y,z,new_roi

print("finish")