from PIL import Image,ImageFilter #Import Pillow library to load the images
import numpy as np #Import numpy
from skimage.morphology import skeletonize,square,dilation #Import functions to compute morphological operations
import os
import pathlib
path_to_save_datasets = "../PVBM_datasets"
from PVBM.Datasets import PVBM_Datasets
import sys
sys.setrecursionlimit(100000)

# dataset_downloader = PVBM_Datasets()
# dataset_downloader.download_dataset("Crop_HRF", path_to_save_datasets)
# dataset_downloader.download_dataset("INSPIRE", path_to_save_datasets)
# dataset_downloader.download_dataset("UNAF", path_to_save_datasets)
# print("Images downloaded successfully")
segmentation_path = list(pathlib.Path(path_to_save_datasets).glob("*/artery/*"))[10]
segmentation = Image.open(segmentation_path) #Open the segmentation
image_path = str(segmentation_path).replace("artery","images").replace("veins", "images")
image = Image.open(image_path)
# Depending on the quality of the segmentation, you would need to regularize (smooth) it more or less
#before computing the skeleton for instance by uncomment the following command
#segmentation = segmentation.filter(ImageFilter.ModeFilter(size=3))

segmentation = np.array(segmentation)/255 #Convert the segmentation to a numpy array with value 0 and 1
skeleton = skeletonize(segmentation)*1 # Compute the skeleton of the segmentation

from PVBM.DiscSegmenter import DiscSegmenter
segmenter = DiscSegmenter()
optic_disc = segmenter.segment(str(segmentation_path).replace("artery","images").replace("veins", "images"))
center, radius, roi, zones_ABC = segmenter.post_processing(optic_disc, max_roi_size = 600)
from PVBM.CentralRetinalAnalysis import CREVBMs
creVBMs = CREVBMs()
zone_A_ = zones_ABC[:,:,1]/255
zone_B_ = zones_ABC[:,:,0]/255
zone_C_ = zones_ABC[:,:,2]/255
outsideB = (zone_C_ - zone_B_)
segmentation_roi = (segmentation * outsideB)
skeleton_roi = (skeleton * outsideB)
out = creVBMs.compute_central_retinal_equivalents(segmentation_roi.copy(), skeleton_roi.copy(),center[0],center[1], radius, artery = True, Toplot = False )
1
