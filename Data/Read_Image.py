from Data.data import Data

from PIL import Image
import os
import os.path


class ReadImage(Data):
    file_path = 'images/'

    def __init__(self, file_path):
        print("Reading image...")
        self.file_path = file_path

    def read_image(self):
        img_list = []
        valid_images = [".jpg", ".png"]
        for file in os.listdir(self.file_path):
            temp_file = os.path.splitext(file)[1]
            if temp_file.lower() not in valid_images:
                continue
            img_list.append(file)

        return img_list
