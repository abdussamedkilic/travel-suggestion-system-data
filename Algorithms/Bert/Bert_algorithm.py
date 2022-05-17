from Algorithms.Bert.bert import Bert


class Bert_algorithm:
    def __init__(self) -> None:
        pass
        
    def test_run(self, sentence1, sentence2 , model , tokenizer):
        # f1 = 'Galata Tower was the tallest building in Constantinople. The upper section of the tower with the conical cap was slightly modified in several restorations during the Ottoman period when it was used as an observation tower for spotting fires'
        # f2 = 'Todays Hagia Sophia is the third building constructed in the same place with a different architectural understanding than its predecessors.'
        bertsimilarity = Bert()
        #model , tokenizer = bertsimilarity.getPretrainedModels("bert-base-uncased")

        dist = bertsimilarity.calculate_distance(sentence1, sentence2 , model , tokenizer)
        return dist
        # print("*"*100, "\n")
        # print('The distance between sentence1: '+sentence1 +
        #       ' and sentence2: '+sentence2+' is '+str(dist))
        # print("*"*100, "\n")
