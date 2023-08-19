import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

input_folder = 'E:\\PET\\Test'  # Folder containing NIfTI files
output_folder = 'E:\\PET\\PNG\\Test'       # Folder to save PNG files

def nifti_to_png(nifti_path, output_folder):
    # Load the NIfTI file
    nifti_image = nib.load(nifti_path)
    nifti_data = nifti_image.get_fdata()
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through the 3D volume and save slices as PNG files
    for slice_idx in range(nifti_data.shape[-1]):
        slice_data = nifti_data[:, :, slice_idx]
        output_path = os.path.join(output_folder, f"{os.path.basename(nifti_path)}_slice_{slice_idx:03d}.png")
        
        # Normalize the slice data to [0, 255] for PNG format
        slice_data_normalized = ((slice_data - np.min(slice_data)) / 
                                 (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)
        
        # Save the slice as a PNG file
        plt.imshow(slice_data_normalized, cmap="gray")
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

# List all files in the input folder
file_list = os.listdir(input_folder)

# Iterate through the files and process NIfTI files
for file_name in file_list:
    if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
        nifti_path = os.path.join(input_folder, file_name)
        nifti_to_png(nifti_path, output_folder)

print("Conversion completed.")

