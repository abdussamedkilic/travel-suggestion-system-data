from regex import D
from Algorithms.CollaborativeFiltering.cosine_similarity import cosine_similarity
from Algorithms.Bert.Bert_algorithm import Bert_algorithm

class CreateMatrix:
    
    def __init__(self):
        print("")
    
    def createScoresMatrix_Doc2vec(self, similarity_results,rateDoc2vec=25):
       
        scores_matrix = [
            [ 0 for j in range(len(similarity_results))] for i in range(0,len(similarity_results))]
        ratedscores_matrix = [
            [ 0 for j in range(len(similarity_results))] for i in range(0,len(similarity_results))]

        for i in range(len(similarity_results)): #place number
            for j in range(len(similarity_results[0])): # top 10 similarity
                # (tag,similarity score)
                scores_matrix[i][similarity_results[i][j][0]] = similarity_results[i][j][1]
                ratedscores_matrix[i][similarity_results[i][j][0]] = similarity_results[i][j][1]*(rateDoc2vec/100)
        
        return [scores_matrix,ratedscores_matrix]
    
    def createScoresMatrix_Cnn(self,feature_vector,rateCnn=50):
        cosine = cosine_similarity()
        score_matrix = [
            [ 0 for j in range(len(feature_vector))] for i in range(len(feature_vector))]
        
        ratedscore_matrix = [
            [ 0 for j in range(len(feature_vector))] for i in range(len(feature_vector))]
        
        for i in range(0,len(feature_vector)):
            for j in range(0,len(feature_vector)):
                score_matrix[i][j] = cosine.find_cnn_image_similarity(feature_vector[i],feature_vector[j])
                ratedscore_matrix[i][j] = cosine.find_cnn_image_similarity(feature_vector[i],feature_vector[j])*(rateCnn/100)
        
        return score_matrix,ratedscore_matrix
    
    def createScoresMatrix_Bert(self,comments,rateBert=25):
        
        bert = Bert_algorithm()
        score_matrix = [
            [ 0 for j in range(len(comments))] for i in range(len(comments))]
        ratedscore_matrix = [
            [ 0 for j in range(len(comments))] for i in range(len(comments))]

        for i in range(0,len(comments)): #len(comments)
              for j in range(0,len(comments)): #len(comments)
                  print("i:"+str(i)+" j:"+str(j))
                  score_matrix[i][j] = bert.test_run(comments[i][0],comments[j][0])
                  ratedscore_matrix[i][j] = bert.test_run(comments[i][0],comments[j][0])*(rateBert/100)
        return [score_matrix,ratedscore_matrix]

    def createScoresMatrix_ImageSimilarity(self,similarity_results):
        print("We aren't using now...")

    def createScoresMatrix_MergedMatrix():
        print()