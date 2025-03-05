import os
import requests
import onnxruntime as ort
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
import cv2

class DiscSegmenter:
    """A class that performs optic disc segmentation."""

    def __init__(self):
        """Initialize the DiscSegmenter class with image size and model path."""
        self.img_size = 512
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, "lunetv2_odc.onnx")
        self.download_model()

    def download_model(self):
        """Download the ONNX model if it does not exist."""
        model_url = 'https://github.com/aim-lab/PVBM/raw/main/PVBM/lunetv2_odc.onnx'
        print(f"Model path: {self.model_path}")
        if not os.path.exists(self.model_path):
            print(f"Downloading model from {model_url}...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f'Model downloaded to {self.model_path}')
        else:
            print("Model already exists, skipping download.")

    def find_biggest_contour(self, contours):
        """
        Find the biggest contour in the provided list of contours.

        :param contours: List of contours.
        :type contours: list of numpy arrays
        :return: The center, radius, and the biggest contour.
        :rtype: tuple
        """
        radius = -1
        final_contour = contours[0]
        (x, y), radius = cv2.minEnclosingCircle(final_contour)
        center = (int(x), int(y))
        radius = int(radius)
        for contour in contours:
            (x, y), radius_ = cv2.minEnclosingCircle(contour)
            if radius_ > radius:
                center = (int(x), int(y))
                radius = int(radius_)
                final_contour = contour
        return center, radius, final_contour

    def post_processing(self, segmentation, max_roi_size):
        """
        Post-process the segmentation result to extract relevant zones.

        :param segmentation: Segmentation result as a numpy array.
        :type segmentation: numpy array
        :param max_roi_size: Maximum size of the region of interest.
        :type max_roi_size: int
        :return: The center, radius, region of interest, and zones ABC.
        :rtype: tuple
        """
        segmentation = np.array(segmentation, dtype=np.uint8)
        try:
            contours, hierarchy = cv2.findContours(segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            center, radius, contour = self.find_biggest_contour(contours)

            one_radius = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
            cv2.circle(one_radius, center, radius, (0, 255, 0), -1)

            two_radius = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
            cv2.circle(two_radius, center, int(radius * 2), (0, 255, 0), -1)

            three_radius = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
            cv2.circle(three_radius, center, radius * 3, (0, 255, 0), -1)

            roi = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
            cv2.circle(roi, center, max_roi_size, (0, 255, 0), -1)

        except:
            one_radius = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
            two_radius = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
            three_radius = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
            roi = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
            center = (segmentation.shape[0] // 2, segmentation.shape[1] // 2)
            radius = 0

        zones_ABC = np.zeros((segmentation.shape[0], segmentation.shape[1], 4))
        zones_ABC[:, :, :3] = one_radius
        zones_ABC[:, :, 0] = two_radius[:, :, 1]
        zones_ABC[:, :, 2] = three_radius[:, :, 1]
        zones_ABC[:, :, 3] = np.maximum(one_radius[:, :, 1], three_radius[:, :, 1]) / 2

        return center, radius, roi, zones_ABC

    def segment(self, image_path):
        """
        Perform the optic disc segmentation given an image path.

        :param image_path: Path to the image.
        :type image_path: str
        :return: A PIL Image containing the Optic Disc segmentation.
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
        od = outputs[0][0, 0] > 0

        return PIL.Image.fromarray(np.array(od, dtype=np.uint8) * 255).resize(original_size, PIL.Image.Resampling.NEAREST)
