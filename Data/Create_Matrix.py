
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

    def createScoresMatrix_Cnn(self,similartiy_results):
        print("")
    
    def createScoresMatrix_Bert(self,similarity_results):
        print("")
    
    def createScoresMatrix_ImageSimilarity(self,similarity_results):
        print("")