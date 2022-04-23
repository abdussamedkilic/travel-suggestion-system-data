from Algorithms.CollaborativeFiltering.cosine_similarity import cosine_similarity

from PIL import Image
import numpy as np


class ImageSimilarity:
    image_list = []

    def __init__(self, image_list):
        print("Image similarity...")
        self.image_list = image_list

    def open_image(self, image_path):
        # image must be "path_name/image_name.jpeg/png"
        # we need to control it. It's format is true or not?
        return Image.open("Images/"+image_path)

    def image_preprocess(self, image):
        image_resized = image.resize((244, 244))
        image_reshape = image_resized.resize(
            (round(image_resized.size[0] * 0.5), round(image_resized.size[1] * 0.5)))

        # convert the images to (R,G,B) arrays
        image_array = np.array(image_reshape)

        # flatten the arrays so they are 1 dimensional vectors
        image_array = image_array.flatten()

        # divide the arrays by 255, the maximum RGB value to make sure every value is on a 0-1 scale
        return image_array / 255

    def implement_image_similarity(self):
        opened_image1 = self.open_image(self.image_list[1])
        opened_image2 = self.open_image(self.image_list[1])
        reshaped_image1 = self.image_preprocess(opened_image1)
        reshaped_image2 = self.image_preprocess(opened_image2)
        return reshaped_image1, reshaped_image2

    def main_image_similarity(self):
        rehaped_image1, rehaped_image2 = self.implement_image_similarity()
        cosine = cosine_similarity()
        # we must be send two image.
        similarity_score = cosine.find_image_similarity(
            rehaped_image1, rehaped_image2)
        print("image similarity score: "+str(similarity_score))
