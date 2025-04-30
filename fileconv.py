import os
import h5py
from PIL import Image
import numpy as np

def convert_images_to_hdf5(input_folder, output_file):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    # Create an HDF5 file
    with h5py.File(output_file, 'w') as hdf5_file:
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(input_folder, image_file)

            # Open and convert the image to a numpy array
            image = np.array(Image.open(image_path))

            # Create a dataset in the HDF5 file
            dataset_name = f'image_{i}'
            hdf5_file.create_dataset(dataset_name, data=image)

if __name__ == "__main__":
    input_folder ="F:/project/archive/indoorCVPR_09/Images/artstudio"
    output_file = "F:/project/archive/indoorCVPR_09/Images/output.h5"

    convert_images_to_hdf5(input_folder, output_file)
