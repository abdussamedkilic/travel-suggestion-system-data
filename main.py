from venv import create
from Algorithms.CollaborativeFiltering.cosine_similarity import cosine_similarity
from Algorithms.ImageSimilarity.Image_Similarity import ImageSimilarity
from Algorithms.Bert.Bert_algorithm import Bert_algorithm
from Algorithms.WordEmbeddings.DoctoVec import DoctoVec
from Algorithms.ImageSimilarity.Cnn import Cnn
from Data.Create_Matrix import CreateMatrix
from Data.Write_Excel import WriteExcel
from Data.Read_Image import ReadImage
from Data.read_file import ReadFile
from Data.Mongo_DB import Mongodb

import numpy as np

isMongodb = True
isDoc2vec = False
isCnn = False
isSaveImage = False  # Read url from mongodb and save in file(part of cnn)
isBert = True
isImageSimilarity = False
isMerge = False  # for merged operation of output results.

city_name = "Istanbul"

# The sum of the values of these variables must be equal 100
rateDoc2Vec = 25
rateCnn = 50
rateBert = 25

if rateDoc2Vec + rateCnn + rateBert != 100:
    raise RuntimeError(
        "The sum of the rate values must be equal 100!!! for Doc2vec,Cnn and Bert Algorithms"
    )

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
     ],  # Bursaé
    [["I like Kocaeli"]  # Kocaeli University
     , ["I love Seka Park"]  # Seka Park
     ]  # Kocaeli
]
"""

create_matrix = CreateMatrix()
write_excel = WriteExcel()

main_mongodb = Mongodb()
doc2vec = DoctoVec([city_name])
readImg = ReadImage("images/" + city_name + "/")
readfl = ReadFile("Output/")
# cosine = cosine_similarity()
# similarity = ImageSimilarity(image_list)

if isMongodb:
    # TODO to Run MONGOGB
    comments, placeName_list, imageUrl_list, detail_list = main_mongodb.read_Mongo_DB(
        city_name
    )
    # TODO to Run Prepare Dataé
    prepread_comments = doc2vec.prepare_comments(comments)  # for doc2vec
    doc2vec.set_doc(prepread_comments)
    doc2vec.set_text_doc(detail_list)

    # just one time.
    # TODO to Run Read Images
    # we will change it to make each city.
    if isSaveImage:
        for i in range(0, len(placeName_list[0])):  # place number
            readImg.read_image_fromURL(imageUrl_list[0][i], placeName_list[0][i])

    # TODO to Run Read All Images :
    image_list = readImg.read_image()

if isDoc2vec:
    # TODO to Run DOC2VEC
    # #similarity_matrix_doc2vec's size = (city number , place number(dynamic for each city))
    similarity_matrix_doc2vec = doc2vec.main_Doc2Vec()  # Recommend from Comments Data

    # TODO to Run Create Scores Matrix
    scoreMatrix_list = (
        []
    )  # scoreMatrix_list's size = (city number , Place Number , Place Number)
    ratedMatrix_list_Doc2vec = []

    result_list = []

    for i in range(0, len(similarity_matrix_doc2vec)):  # city number
        result_list.append(
            create_matrix.createScoresMatrix_Doc2vec(similarity_matrix_doc2vec[i])
        )
        scoreMatrix_list.append(result_list[i][0])
        ratedMatrix_list_Doc2vec.append(result_list[i][1])

    # TODO to Run Write Matrix to Excel
    for i in range(0, len(scoreMatrix_list)):  # city number
        # placeName_list's size(1,place number) --> just one city for places name. we will change.
        write_excel.writeExcel_Doc2vec(scoreMatrix_list[i], placeName_list, city_name)
        write_excel.writeExcel_Doc2vec(
            ratedMatrix_list_Doc2vec[i], placeName_list, city_name + "_rated"
        )
    write_excel.workbook_doc2vec.close()

if isCnn:
    # TODO to Run CNN
    cnn = Cnn(image_list, city_name)
    feature_vector = cnn.main_Cnn()  # feature_vector is a list. size = (1,image number)

    # TODO to Run Create Scores Matrix
    similarity_score, rated_similarityScore_cnn = create_matrix.createScoresMatrix_Cnn(
        feature_vector, rateCnn
    )
    # rated_similarityScore_cnn = create_matrix.createRatedScoresMatrix_Cnn(feature_vector,rateCnn)

    # TODO to Run Write Matrix to Excel
    write_excel.writeExcel_CNN(
        image_list, city_name, similarity_score
    )  # just one city for places name. we will change.
    write_excel.writeExcel_CNN(
        image_list, city_name + "_rated", rated_similarityScore_cnn
    )
    write_excel.workbook_cnn.close()

if isBert:
    # TODO to Run Bert
    # prepread_comments's size = (city number , place number)

    # TODO to Run Create Scores Matrix
    scoreMatrix , ratedScore_matrix = create_matrix.createScoresMatrix_Bert(comments,rateBert)
    
    # TODO to Run Write Matrix to Excel
    write_excel.writeExcel_Bert(scoreMatrix, placeName_list, city_name)
    write_excel.writeExcel_Bert(
            ratedScore_matrix, placeName_list, city_name + "_rated"
        )

    write_excel.workbook_bert.close()

    """
    score_matrix_List = []  # list for city
    rated_scoreMatrix_list_bert = []

    result_list = []

    for i in range(0, len(prepread_comments)):  # city number
        result_list.append(create_matrix.createScoresMatrix_Bert(prepread_comments[i]))
        score_matrix_List.append(result_list[i][0])
        rated_scoreMatrix_list_bert.append(result_list[i][1])

    # TODO to Run Write Matrix to Excel
    for i in range(0, len(score_matrix_List)):
        write_excel.writeExcel_Bert(score_matrix_List[i], placeName_list, city_name)
        write_excel.writeExcel_Bert(
            rated_scoreMatrix_list_bert[i], placeName_list, city_name + "_rated"
        )
    write_excel.workbook_bert.close()
    """

if isImageSimilarity:
    print("We are not using now...")
    # ! warning, that's not work for now
    # ? Image Similarity :
    # similarity.main_image_similarity()

if isMerge:

    # TODO to Run Read Output Results
    df_doc2vec = readfl.Read_Excel_Rated("Doc2vec_output.xlsx", city_name)
    doc2vec_places = list(df_doc2vec.columns)
    doc2vec_Scorematrix = np.array(df_doc2vec)

    df_cnn = readfl.Read_Excel_Rated("CNN_output.xlsx", city_name)
    cnn_places = list(df_cnn.columns)
    cnn_Scorematrix = np.array(df_cnn)

    df_bert = readfl.Read_Excel_Rated("Bert_output.xlsx", city_name)
    bert_places = list(df_bert.columns)
    bert_Scorematrix = np.array(df_bert)

    # TODO to Run Create Score Matrix
    merged_matrix = create_matrix.createScoresMatrix_MergedResults(
        doc2vec_Scorematrix,
        cnn_Scorematrix,
        bert_Scorematrix,
        doc2vec_places,
        cnn_places,
    )

    # TODO to Run Write Matrix to Excel
    write_excel.writeExcel_MergedResult(merged_matrix, doc2vec_places, city_name)
    write_excel.workbook_merged.close()
