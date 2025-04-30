import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import h5py
from tqdm import tqdm
import skimage.transform as sktrans
import matplotlib.pyplot as plt

# Step 1: Read JPG Images and XML Annotations
jpg_folder = "F:/project/archive/indoorCVPR_09/Images/tv_studio"
xml_folder = "F:/project/archive/indoorCVPR_09annotations/Annotations/tv_studio"
jpg_files = [os.path.join(jpg_folder, file) for file in os.listdir(jpg_folder) if file.endswith(".jpg")]
xml_files = [os.path.join(xml_folder, file) for file in os.listdir(xml_folder) if file.endswith(".xml")]

# Step 2: Modify the Data Loading Section
def get_processed_image(jpg_path):
    # Load JPG image using OpenCV
    im = cv2.imread(jpg_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    im = im / 255.0  # Normalize to [0, 1]
    
    # Add additional processing if needed

    return im

# Step 4: Generate Hazy Images
def generate_hazy_image(im, im_depth, airlight, scatt_coeff):
    # Assuming im_depth is provided in the XML annotations
    trans_map = np.exp(-scatt_coeff * im_depth)
    t = trans_map.reshape(im.shape[0], im.shape[1], 1)
    hazy_im = t * im + (1 - t) * airlight.reshape(1, 1, 3)
    return hazy_im, trans_map

# Step 5: Generate Hazy Dataset
def generate_hazy_dataset(jpg_files, xml_files, output_path, num_airlights, num_scats,
                          a_high=1.0, a_low=0.7, scatt_high=1.2, scatt_low=0.5):
    num_samples = len(jpg_files)

    # prepare output file
    output_file = h5py.File(output_path, "w")
    output_file.create_dataset("airlight", (num_samples,), dtype="float32")
    output_file.create_dataset("scatt_coeff", (num_samples,), dtype="float32")
    output_file.create_dataset("image", (num_samples, 240, 320, 3), dtype="float32")
    output_file.create_dataset("hazy_image", (num_samples, 240, 320, 3), dtype="float32")
    output_file.create_dataset("trans_map", (num_samples, 240, 320), dtype="float32")

    for s_idx in tqdm(range(num_samples)):
        jpg_path = jpg_files[s_idx]
        xml_path = xml_files[s_idx]

        # Load JPG image
        im = get_processed_image(jpg_path)

        # Load XML annotation for depth information
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Extract depth information as needed; adjust this based on your XML structure
        im_depth = np.zeros((im.shape[0], im.shape[1]))  # Replace with actual depth extraction

        output_file["image"][s_idx] = im

        # Generate hazy images
        for j in range(num_airlights):
            airlight = np.zeros(3) + (a_high - a_low) * np.random.random() + a_low
            output_file["airlight"][s_idx] = airlight[0]
            
            for k in range(num_scats):
                scatt_coeff = (scatt_high - scatt_low) * np.random.random() + scatt_low
                output_file["scatt_coeff"][s_idx] = scatt_coeff

                hazy_im, trans_map = generate_hazy_image(im, im_depth, airlight, scatt_coeff)
                output_file["hazy_image"][s_idx] = hazy_im
                output_file["trans_map"][s_idx] = trans_map

    output_file.close()

# Step 6: Check the Results
# Visualize some hazy images and their corresponding transmission maps if needed

# Run the code
NUM_TEST = 40
trn_output_path = "F:/project/archive/indoorCVPR_09/Images/tv_studio/nyu_hazy_trn.mat"
tst_output_path = "F:/project/archive/indoorCVPR_09/Images/tv_studio/nyu_hazy_tst.mat"
shuffle = np.random.permutation(len(jpg_files))
trn_idcs = shuffle[NUM_TEST:]
tst_idcs = shuffle[:NUM_TEST]

generate_hazy_dataset([jpg_files[i] for i in trn_idcs], [xml_files[i] for i in trn_idcs], trn_output_path, 1, 3)
generate_hazy_dataset([jpg_files[i] for i in tst_idcs], [xml_files[i] for i in tst_idcs], tst_output_path, 1, 3)

# Continue with the rest of your code for checking the results, plotting, and cleanup
