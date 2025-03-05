import gdown
import os
import zipfile


class PVBMDataDownloader:
    """A class that downloads the PVBM datasets."""
    def __init__(self):
        self.file_ids = {
            "UNAF": "1IM5qUEARNp2RFpzKmILbdgasjLuJEIcX",
            "INSPIRE": "18TcmkuN_eZgM2Ph5XiX8x7_ejtKhA3qb",
            "Crop_HRF": "1QcozuK5yDyXbBkHqkbM5bxEkTGzkPDl3"
        }

    def download_dataset_from_google_drive(self, file_id, save_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, save_path, quiet=False)
        print(f"Dataset downloaded and saved to {save_path}")

    def unzip_file(self, zip_file_path, extract_to_dir):
        """
        Unzip a file to the specified directory and delete the zip file after extraction.

        :param zip_file_path: Path to the zip file.
        :type zip_file_path: String

        :param extract_to_dir: Directory where the contents will be extracted.
        """
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        print(f"Files extracted to {extract_to_dir}")

        os.remove(zip_file_path)
        print(f"Deleted the zip file: {zip_file_path}")

    def download_dataset(self, name, save_folder_path):
        """
        Download the PVBM datasets

        :param name: Name of the dataset to download. Need to be within {"UNAF", "INSPIRE", "Crop_HRF"}
        :param save_folder_path: Path to the folder where to store the downloaded datasets
        """
        if name in list(self.file_ids.keys()):
            zip_save_path = f"{name}.zip"
            os.makedirs(save_folder_path, exist_ok=True)
            # Call the function to download the dataset
            self.download_dataset_from_google_drive(self.file_ids[name], zip_save_path)
            # Call the function to unzip the file and delete the zip file
            self.unzip_file(zip_save_path, save_folder_path)
        else:
            print("Name should be within the following:", list(self.file_ids.keys()))