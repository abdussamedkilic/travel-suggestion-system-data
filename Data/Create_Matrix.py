from Algorithms.CollaborativeFiltering.cosine_similarity import cosine_similarity

class CreateMatrix:
    
    def __init__(self):
        print("")
    
    def createScoresMatrix_Doc2vec(self, similarity_results):
       
        scores_matrix = [
            [ 0 for j in range(len(similarity_results))] for i in range(0,len(similarity_results))]

        for i in range(len(similarity_results)): #place number
            for j in range(len(similarity_results[0])): # top 10 similarity
                # (tag,similarity score)
                scores_matrix[i][similarity_results[i][j][0]] = similarity_results[i][j][1]
        
        return scores_matrix

    def createScoresMatrix_Cnn(self,feature_vector):
        cosine = cosine_similarity()
        score_matrix = [
            [ 0 for j in range(len(feature_vector))] for i in range(len(feature_vector))]

        for i in range(0,len(feature_vector)):
            for j in range(0,len(feature_vector)):
                score_matrix[i][j] = cosine.find_cnn_image_similarity(feature_vector[i],feature_vector[j])
        
        return score_matrix
    
    def createScoresMatrix_Bert(self,similarity_results):
        print("")
    
    def createScoresMatrix_ImageSimilarity(self,similarity_results):
        print("")