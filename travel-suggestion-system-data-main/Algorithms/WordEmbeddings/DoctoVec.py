
from Algorithms.WordEmbeddings.word_embedding import word_embedding
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

import re
import nltk
from nltk.stem import WordNetLemmatizer

"""
**** You must be just execute one more time that code.

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
"""

stemmer = WordNetLemmatizer()
en_stop = set(nltk.corpus.stopwords.words('english'))


class DoctoVec(word_embedding):
    """
    https://thinkinfi.com/gensim-doc2vec-python-implementation/
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
    https://stackabuse.com/python-for-nlp-working-with-facebook-fasttext-library/

    dm = 0 pv-dbow
    dm = 1 pv-dm

    """

    city_name = ["Istanbul_deneme", "Bursa_deneme", "Kocaeli_deneme"]

    doc = []
    text_doc = []
    dm = 0
    want_train = True

    def __init__(self, doc, text_doc):

        self.doc = doc
        self.text_doc = text_doc

    def find_dimension(self, matrix):

        """
        matrix must be numpy.array or list.
        """
        # row number , column number
        return len(matrix), len(matrix[0])

    def preprocesses(self, sentences):

        """
        Sadece 1 tane cümle veriyoruz ve o bize cümleyi düzeltip, text olarak geri gönderiyor.
        Örneğin:
        sentence = ['I love Galata Tower, I like Istanbul']
        preprocess_text = love galata tower like istanbul

        """

        # Remove all the special characters
        sentences = re.sub(r'\W', ' ', str(sentences))

        # remove all single characters
        sentences = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentences)

        # Remove single characters from the start
        sentences = re.sub(r'\^[a-zA-Z]\s+', ' ', sentences)

        # Substituting multiple spaces with single space
        sentences = re.sub(r'\s+', ' ', sentences, flags=re.I)

        # Removing prefixed 'b'
        sentences = re.sub(r'^b\s+', '', sentences)

        # Converting to Lowercase
        sentences = sentences.lower()

        # Lemmatization
        tokens = sentences.split()
        # tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if
                  len(word) > 3]  # if your word's length smaller than 3, we eleminated that word

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    def implementAlgorithm(self, document):

        preprocess_document = [[[] for j in range(len(document[0]))] for i in range(len(document))]
        tokenizer_document = [[[] for j in range(len(document[0]))] for i in range(len(document))]
        tagged_document = [[] for i in range(len(document))]
        # deneme_tagged_document = []
        model_list = []
        result_similarity_list = []

        # step 1 - Preprocess
        for i in range(0, len(document)):  # City Number
            for j in range(0, len(document[0])):  # Places Number of a city.
                preprocess_document[i][j].append(self.preprocesses(document[i][j]))

        print("after preprocess\n" + str(preprocess_document))

        # step 2 - Word Tokenizer
        for i in range(0, len(preprocess_document)):
            for j in range(0, len(preprocess_document[0])):
                tokenizer_document[i][j] = self.word_tokenizer(preprocess_document[i][j])

        print("tokenizer_document\n" + str(tokenizer_document))
        print("Tokenizer Document Size:" + str(self.find_dimension(tokenizer_document)))
        print(tokenizer_document[0][1])

        # step 3 - Tagging Document

        for i in range(0, len(tokenizer_document)):
            tagged_document[i] = self.tagged_document(tokenizer_document[i])

        print("tagged document \n" + str(tagged_document))
        print("Tagged Document Size:" + str(self.find_dimension(tagged_document)))

        # step 4 - Train Doc2Vec Model

        for i in range(0, len(tagged_document)):

            if self.want_train == True:
                model_doc2vec = self.train_doc2vec(tagged_document[i])
                # model_doc2vec= model_doc2vec.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
                self.save_model(model_doc2vec, self.city_name[i])  # just train one more time.

            model_list.append(self.load_model("Models/" + self.city_name[i] + "_doc2vec.model"))

        # step 5 - Find Similarity Matrix
        for i in range(0, len(model_list)):
            result_similarity_list.append(self.find_similarity_to_matrix(model_list[i], "I love Bursa"))

        print("\nResults Similarity")
        print(result_similarity_list)


    def preprocesses(self, sentences):

        """
        Sadece 1 tane cümle veriyoruz ve o bize cümleyi düzeltip, text olarak geri gönderiyor.
        Örneğin:
        sentence = ['I love Galata Tower, I like Istanbul']
        preprocess_text = love galata tower like istanbul

        """

        # Remove all the special characters
        sentences = re.sub(r'\W', ' ', str(sentences))

        # remove all single characters
        sentences = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentences)

        # Remove single characters from the start
        sentences = re.sub(r'\^[a-zA-Z]\s+', ' ', sentences)

        # Substituting multiple spaces with single space
        sentences = re.sub(r'\s+', ' ', sentences, flags=re.I)

        # Removing prefixed 'b'
        sentences = re.sub(r'^b\s+', '', sentences)

        # Converting to Lowercase
        sentences = sentences.lower()

        # Lemmatization
        tokens = sentences.split()
        # tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if
                  len(word) > 3]  # if your word's length smaller than 3, we eleminated that word

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    def word_tokenizer(self, sentence):

        """
        sentence = ["I love istanbul], sentence[0] = "I love istanbul"
        We need to just a list.
        sentence's size must be = (1,1)
        """
        return word_tokenize(sentence[0].lower())

    def tagged_document(self, tokenized_list):
        """
        --> tokenized_list must be:

        [[‘i’, ‘love’, ‘data’, ‘science’],
        [‘i’, ‘love’, ‘coding’, ‘in’, ‘python’],
        [‘i’, ‘love’, ‘building’, ‘nlp’, ‘tool’],
        [‘this’, ‘is’, ‘a’, ‘good’, ‘phone’],
        [‘this’, ‘is’, ‘a’, ‘good’, ‘tv’],
        [‘this’, ‘is’, ‘a’, ‘good’, ‘laptop’]]

        --> You should send to The Places of Istanbul. so just a city
        """

        # Convert tokenized document into gensim formated tagged data
        return [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_list)]

    def train_doc2vec(self, tagged_document):

        """
        tagged_document:

        [TaggedDocument(words=[‘i’, ‘love’, ‘data’, ‘science’], tags=[0]),
         TaggedDocument(words=[‘i’, ‘love’, ‘coding’, ‘in’, ‘python’], tags=[1]),
         TaggedDocument(words=[‘i’, ‘love’, ‘building’, ‘nlp’, ‘tool’], tags=[2]),
         TaggedDocument(words=[‘this’, ‘is’, ‘a’, ‘good’, ‘phone’], tags=[3]),
         TaggedDocument(words=[‘this’, ‘is’, ‘a’, ‘good’, ‘tv’], tags=[4]),
         TaggedDocument(words=[‘this’, ‘is’, ‘a’, ‘good’, ‘laptop’], tags=[5])]

         --> You should send to The Places of Istanbul. so just a city

        """

        return Doc2Vec(tagged_document, vector_size=15, dm=self.dm, window=2, min_count=1, workers=4, epochs=200)

    def train_model(self, tagged_document):

        print("model is training...")

    def save_model(self, model, model_name):

        print("Saving model...")
        model.save("Models/" + model_name + "_doc2vec.model")
        print("Model is saved")

    def load_model(self, model_path):
        print("Loading model...")
        return Doc2Vec.load(model_path)

    def find_similarity_to_matrix(self, model, test_sentence):
        """
        test_sentence must be a String.
        test_sentence = "That is a good device"

        """

        # find most similar doc
        test_doc = word_tokenize(test_sentence.lower())
        return model.docvecs.most_similar(positive=[model.infer_vector(test_doc)], topn=10)

    def main_Doc2Vec(self):

        """
        first step:  create instance of Doc2Vec with constructor
        second step: call the main_Doc2Vec
        """

        self.implementAlgorithm(self.doc)






