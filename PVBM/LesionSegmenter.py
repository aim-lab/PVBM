import os
import requests
import onnxruntime as ort
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
import cv2

class LesionSegmenter:
    """A class that performs lesion segmentation."""

    def __init__(self, model_path):
        """Initialize the LesionSegmenter class with image size and model path.
           Download the model from: https://drive.google.com/file/d/1pOhp506d0jiUzOWJBLoH6yXagkMLb8-t/view?usp=sharing
        """
        self.img_size = 512
        script_dir = os.path.dirname(os.path.abspath(__file__))
        #self.model_path = os.path.join(script_dir, "lesion.onnx")
        self.model_path = model_path


    def segment(self, image_path):
        """
        Perform the lesion segmentation given an image path.

        :param image_path: Path to the image.
        :type image_path: str
        :return: A PIL Image containing the Lesion segmentation.
        :rtype: PIL.Image
        """
        session = ort.InferenceSession(self.model_path)
        input_name = session.get_inputs()[0].name

        img_orig = PIL.Image.open(image_path)
        original_size = img_orig.size
        image = img_orig.resize((self.img_size, self.img_size))
        image = transforms.ToTensor()(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        image_np = image.numpy()
        image_np = np.expand_dims(image_np, axis=0)

        outputs = session.run(None, {input_name: image_np})
        lesion = self.visualize_lesion(outputs[0])

        return PIL.Image.fromarray(lesion).resize(original_size, PIL.Image.Resampling.NEAREST)

    def visualize_lesion(self,pred):
        # Initialize an empty RGBA image
        visualisation = np.zeros((pred.shape[-2], pred.shape[-1], 4), dtype=np.uint8)

        # Create masks for each variable where the pixel values are greater than zero
        ex_mask = pred[0, 0] > 0
        se_mask = pred[0, 1] > 0
        he_mask = pred[0, 2] > 0
        ma_mask = pred[0, 3] > 0

        # Assign colors to each variable
        # Pink color for 'se' (RGB: 255, 192, 203)
        visualisation[se_mask, 0] = 255  # Red channel
        visualisation[se_mask, 1] = 192  # Green channel
        visualisation[se_mask, 2] = 203  # Blue channel

        # White color for 'he' (RGB: 255, 255, 255)
        visualisation[he_mask, 0] = 255  # Red channel
        visualisation[he_mask, 1] = 255  # Green channel
        visualisation[he_mask, 2] = 255  # Blue channel

        # Cyan color for 'ma' (RGB: 0, 255, 255)
        visualisation[ma_mask, 0] = 0  # Red channel
        visualisation[ma_mask, 1] = 255  # Green channel
        visualisation[ma_mask, 2] = 255  # Blue channel

        # Orange color for 'ex' (RGB: 255, 165, 0)
        visualisation[ex_mask, 0] = 255  # Red channel
        visualisation[ex_mask, 1] = 165  # Green channel
        visualisation[ex_mask, 2] = 0  # Blue channel

        # Set the alpha channel to make the colors visible where any variable is present
        alpha_mask = se_mask | he_mask | ma_mask | ex_mask
        visualisation[alpha_mask, 3] = 255  # Alpha channel
        return visualisation
