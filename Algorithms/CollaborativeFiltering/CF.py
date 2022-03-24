from Algorithms.CollaborativeFiltering.collaborative_filtering import collaborative_filtering


class CF(collaborative_filtering):

    def __init__(self):
        print("constructor")

    def recommender(self):
        print("Result matrix")
    def findSimilarity(self):
        print("We will use the cosine or other similarity methods")
    def findObject(self):
        print("We will use the KNN algorithm or other algorithms")


