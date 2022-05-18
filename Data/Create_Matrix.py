from Algorithms.CollaborativeFiltering.cosine_similarity import cosine_similarity
from Algorithms.Bert.Bert_algorithm import Bert_algorithm
from Algorithms.Bert.bert import Bert

import numpy as np


class CreateMatrix:
    def __init__(self) -> None:
        pass

    def createScoresMatrix_Doc2vec(self, similarity_results, rateDoc2vec=25):

        scores_matrix = [
            [0 for j in range(len(similarity_results))]
            for i in range(0, len(similarity_results))
        ]
        ratedscores_matrix = [
            [0 for j in range(len(similarity_results))]
            for i in range(0, len(similarity_results))
        ]

        for i in range(len(similarity_results)):  # place number
            for j in range(len(similarity_results[0])):  # top 10 similarity
                # (tag,similarity score)
                scores_matrix[i][similarity_results[i][j][0]] = similarity_results[i][
                    j
                ][1]
                ratedscores_matrix[i][similarity_results[i][j][0]] = similarity_results[
                    i
                ][j][1] * (rateDoc2vec / 100)

        return [scores_matrix, ratedscores_matrix]

    def createScoresMatrix_Cnn(self, feature_vector, rateCnn=50):
        cosine = cosine_similarity()
        score_matrix = [
            [0 for j in range(len(feature_vector))] for i in range(len(feature_vector))
        ]

        ratedscore_matrix = [
            [0 for j in range(len(feature_vector))] for i in range(len(feature_vector))
        ]

        for i in range(0, len(feature_vector)):
            for j in range(0, len(feature_vector)):
                score_matrix[i][j] = cosine.find_cnn_image_similarity(
                    feature_vector[i], feature_vector[j]
                )
                ratedscore_matrix[i][j] = cosine.find_cnn_image_similarity(
                    feature_vector[i], feature_vector[j]
                ) * (rateCnn / 100)

        return score_matrix, ratedscore_matrix

    def createScoresMatrix_Bert(self, comments, rateBert=25):

        bert = Bert_algorithm()
        bertsimilarity = Bert()
        model , tokenizer = bertsimilarity.getPretrainedModels("bert-base-uncased")

        #score_matrix's size = (place number , place number) for each city
        score_matrix = [
            [0 for j in range(len(comments))] for i in range(len(comments))]
        ratedscore_matrix = [
            [0 for j in range(len(comments))] for i in range(len(comments))
        ]

        for i in range(0,len(comments)): # place number len(comments)
            for j in range(0,len(comments)): # place number len(comments)
                print("row ",i," column ",j)
                result_temp = []
                for k in range(0,len(comments[0])): # comment number in a place --> len(comments[0])
                    for a in range(0,len(comments[0])): # comment number in a place --> len(comments[0])
                        result_temp.append(bert.test_run(comments[i][k],comments[j][a] , model, tokenizer))
                
                if i == j:
                     score_matrix[i][j] = np.max(result_temp)
                else:
                     score_matrix[i][j] = np.min(result_temp)
                
                ratedscore_matrix[i][j] = score_matrix[i][j]*(rateBert/100)
                # score_matrix[i][j] = np.mean(result_temp)
                # ratedscore_matrix[i][j] = np.mean(result_temp)*(rateBert/100)
                    
        return [score_matrix, ratedscore_matrix]           

    def createScoresMatrix_ImageSimilarity(self, similarity_results):
        print("We aren't using now...")

    def createScoresMatrix_MergedResults(
        self,
        doc2vec_matrix,
        cnn_matrix,
        bert_matrix,
        placeName_list,
        cnn_placeName_list,
    ):

        merged_matrix = [
            [0 for j in range(len(doc2vec_matrix))] for i in range(len(doc2vec_matrix))
        ]

        for i in range(0, len(doc2vec_matrix)):  # placeNumber
            rowIndex_Cnn = cnn_placeName_list.index(placeName_list[i])
            for j in range(0, len(doc2vec_matrix)):  # placeNumber
                colIndex_Cnn = cnn_placeName_list.index(placeName_list[j])
                merged_matrix[i][j] = (
                    doc2vec_matrix[i][j]
                    + cnn_matrix[rowIndex_Cnn][colIndex_Cnn]
                    + bert_matrix[i][j]
                )

        return merged_matrix
