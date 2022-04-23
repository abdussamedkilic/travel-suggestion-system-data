from scipy import spatial
from skimage.transform import resize


class cosine_similarity:
    def __init__(self):
        print("Cosine Similarity...")

    def find_image_similarity(self, image1, image2):
        shape = image1.shape
        img2_resized = resize(image2, shape)
        img1_vector = image1.ravel()
        img2_vector = img2_resized.ravel()
        return 1-spatial.distance.cosine(img1_vector, img2_vector)

    def find_cnn_image_similarity(self, vector1, vector2):
        sim_cos = 1 - spatial.distance.cosine(vector1, vector2)
        return sim_cos
