from skimage.transform import resize
from scipy.spatial.distance import cosine


class cosine_similarity:
    def __init__(self):
        print("Cosine Similarity...")

    def find_image_similarity(self, image1, image2):
        shape = image1.shape
        img2_resized = resize(image2, shape)
        img1_vector = image1.ravel()
        img2_vector = img2_resized.ravel()
        return 1 - cosine(img1_vector, img2_vector)

    def find_cnn_image_similarity(self, vector1, vector2):
        return 1 - cosine(vector1, vector2)

    def calculate_bert_sentence_distance(sentence_1, sentence_2):
        return 1 - cosine(sentence_1, sentence_2)
