from Algorithms.CollaborativeFiltering.cosine_similarity import cosine_similarity
from Algorithms.ImageSimilarity.Image_Similarity import ImageSimilarity
from Algorithms.Bert.Bert_algorithm import Bert_algorithm
from Algorithms.WordEmbeddings.DoctoVec import DoctoVec
from Algorithms.ImageSimilarity.Cnn import Cnn
from Data.Create_Matrix import CreateMatrix
from Data.Write_Excel import WriteExcel
from Data.Read_Image import ReadImage
from Data.Mongo_DB import Mongodb

isMongodb = True
isDoc2vec = False
isCnn = False
isBert = True
isImageSimilarity = False
isSaveImage = False # Read url from mongodb and save in file.

city_name = "Istanbul"

"""
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
""" 


create_matrix = CreateMatrix()
write_excel = WriteExcel()

main_mongodb = Mongodb()
doc2vec = DoctoVec([city_name])
readImg = ReadImage('images/'+city_name+"/")
#cosine = cosine_similarity()
#similarity = ImageSimilarity(image_list)

if isMongodb:
    # TODO to Run MONGOGB
    comments,placeName_list,imageUrl_list,detail_list = main_mongodb.read_Mongo_DB(city_name)
    # TODO to Run Prepare Data
    prepread_comments = doc2vec.prepare_comments(comments) # for doc2vec
    doc2vec.set_doc(prepread_comments)
    doc2vec.set_text_doc(detail_list)

    # just one time.
    # TODO to Run Read Images
    # we will change it to make each city.
    if isSaveImage:
        for i in range(0,len(placeName_list[0])): # place number
            readImg.read_image_fromURL(imageUrl_list[0][i],placeName_list[0][i])
    
    # TODO to Run Read All Images :
    image_list = readImg.read_image()
    #print("read image list:\n"+str(image_list))

if isDoc2vec:
    # TODO to Run DOC2VEC
    similarity_matrix_doc2vec = doc2vec.main_Doc2Vec()# Recommend from Comments Data
    #Similarity_matrix_doc2vec's size = (city number , place number(dynamic for each city)) 

    # TODO to Run Create Scores Matrix
    scoreMatrix_list = []
    for i in range(0,len(similarity_matrix_doc2vec)): #city number
        scoreMatrix_list.append(create_matrix.createScoresMatrix_Doc2vec(similarity_matrix_doc2vec[i]))
    # scoreMatrix_list's size = (city number , Place Number , Place Number)

    # TODO to Run Write Matrix to Excel
    for i in range(0,len(scoreMatrix_list)): # city number
        # placeName_list's size(1,place number) --> just one city for places name. we will change.
        write_excel.writeExcel_Doc2vec(scoreMatrix_list[i],placeName_list,city_name) 
               
if isCnn:
    # TODO to Run CNN 
    cnn = Cnn(image_list,city_name)
    feature_vector = cnn.main_Cnn()  # feature_vector is a list. size = (1,image number)
    
    similarity_score = create_matrix.createScoresMatrix_Cnn(feature_vector)
    write_excel.writeExcel_CNN(image_list,city_name,similarity_score)  #just one city for places name. we will change.
    

if isBert:
    # TODO to Run Bert
    #prepread_comments's size = (city number , place number)
    score_matrix_List = [] # list for city 
   
    for i in range(0,len(prepread_comments)): # city number
         score_matrix_List.append(create_matrix.createScoresMatrix_Bert(prepread_comments[i]))

    for i in range(0,len(score_matrix_List)):
        write_excel.writeExcel_Bert(score_matrix_List[i],placeName_list,city_name)    
    
    #print("score matrix:\n"+str(score_matrix_List[0]))

    #Bert_algorithm.test_run()
if isImageSimilarity:
    print("We are not using now...")
    # ! warning, that's not work for now
    # ? Image Similarity :
    # similarity.main_image_similarity()

