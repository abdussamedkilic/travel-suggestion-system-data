from Algorithms.CollaborativeFiltering.Cosine_Similarity import cosine_similarity
from Algorithms.ImageSimilarity.Image_Similarity import ImageSimilarity
from Algorithms.WordEmbeddings.DoctoVec import DoctoVec
from Algorithms.ImageSimilarity.Cnn import Cnn
from Data.Read_Image import ReadImage


document = [
    [["I love Galata Tower, I like Istanbul"]  # Galata Kulesi,"comment1 , comment2"
     , ["I don't like Maiden's Tower. I like Istanbul"]  # Kız Kulesi

     ],  # İstanbul
    [["I loved this place. There are so many beautiful shops and you can buy anything you want."]  # Kapalı Çarşı
        , ["I love Ulu Mosque, I like Bursa"]  # Ulu cami
     ],  # Bursa
    [
        ["I love Kocaeli University, I don't like Kocaeli"]  # Kocaeli Universitesi
        , ["I love Seka Park, I like Kocaeli"]  # Seka Park
    ]  # Kocaeli
]

test_document = [
    [["I love Istanbul"]  # Galata Kulesi Describiton
     , ["I like Maiden's Tower"]  # Kız Kulesi
     ],  # Istanbul
    [["I hate shopping this place.Beacuse there are a lot of people everywhere"]  # Kapali Carsi
     , ["I love this place.When ı go the Mosque, ı feel like peaceful"]  # Ulu Cami
     ],  # Bursa
    [["I like Kocaeli"]  # Kocaeli University
     , ["I love Seka Park"]  # Seka Park
     ]  # Kocaeli
]


# TODO : To Run DOC2VEC
# Recommend from Comments Data
# doc2vec = DoctoVec(document , test_document)
# similarity_matrix_doc2vec=doc2vec.main_Doc2Vec()
# print("\similarity_matrix_doc2vec Similarity")
# print(similarity_matrix_doc2vec)


readImg = ReadImage('images/')
image_list = readImg.read_image()
print("read image list:\n"+str(image_list))

cosine = cosine_similarity()

# ! warning, that's not work for now
#similarity = ImageSimilarity(image_list)
# similarity.main_image_similarity()

# ? CNN SIDE :
cnn = Cnn(image_list)
feature_vector = cnn.main_Cnn()  # feature_vector is a list. size = (1,image number)
similarity_score = cosine.find_cnn_image_similarity(
    feature_vector[1], feature_vector[2])
print("Cnn similarity score:"+str(similarity_score))
