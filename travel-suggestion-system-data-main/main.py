
"""
Python version is 3.9.4. You must be select version 3.9 or upper.
Environment is PyCharm of Anaconda Editor.

"""
from Algorithms.WordEmbeddings.DoctoVec import DoctoVec

document = [
   [["I love Galata Tower, I like Istanbul"]#Galata Kulesi,"cm1 , cm2"
    ,["I don't like Maiden's Tower. I like Istanbul"]#Kız Kulesi

   ],#İstanbul
   [["I loved this place. There are so many beautiful shops and you can buy anything you want."]#Kapalı Çarşı
    ,["I love Ulu Mosque, I like Bursa"]#Ulu cami
    ],#Bursa
   [
        ["I love Kocaeli University, I don't like Kocaeli"]#Kocaeli Universitesi
        ,["I love Seka Park, I like Kocaeli"]#Seka Park
    ]#Kocaeli
]

test_document = [

    [["I love Istanbul"]#Galata Kulesi Describiton
     ,["I like Maiden's Tower"]#Kız Kulesi
    ],#Istanbul
    [["I hate shopping this place.Beacuse there are a lot of people everywhere"]#Kapali Carsi
     ,["I love this place.When ı go the Mosque, ı feel like peaceful"]#Ulu Cami
    ],#Bursa
    [["I like Kocaeli"]#Kocaeli University
     ,["I love Seka Park"]#Seka Park
    ]#Kocaeli
]

#print(""+str(len(document))+" "+str(len(document[0])))

example = DoctoVec(document , test_document)
example.main_Doc2Vec()


