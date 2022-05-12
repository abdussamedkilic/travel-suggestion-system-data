# from Algorithms.CollaborativeFiltering.cosine_similarity import cosine_similarity
import os
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16

# from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input

"""
ref:https://github.com/lbrejon/Compute-similarity-between-images-using-CNN/blob/main/notebooks/extract_features_models.ipynb

"""


class Cnn:
    image_list = []
    img_path = "images/"
    available_models = ["vgg16", "resnet50"]  # Transfer Learning methods
    selected_model = "vgg16"

    def __init__(self, image_list, city_name):
        self.image_list = image_list
        self.img_path += city_name + "/"

    def load_model(self, model_name, include_top=True):
        """Load pre-trained Keras model
        Args:
            model_name: String, name of model to load
            include_top: String, the model is buildt with 'feature learning block' + 'classification block'
            Returns: model: Keras model instance
        """
        if self.selected_model in self.available_models:
            # Load a Keras instance
            try:
                if model_name == "vgg16":
                    model = VGG16(weights="imagenet", include_top=include_top)
                elif model_name == "resnet50":
                    print("")
                    # model = ResNet50(weights='imagenet',
                    #                  include_top=include_top)
                print(f">> '{model.name}' model successfully loaded!")
            except:
                print(f">> Error while loading model '{self.selected_model}'")
                raise RuntimeError("Error while loading model")

        # Wrong selected model
        else:
            print(
                f">> Error: there is no '{self.selected_model}' in {self.available_models}"
            )
            raise RuntimeError(
                "Error: there is no '{self.selected_model}' in {self.available_models}"
            )

        return model

    def get_img_size_model(self, model):
        """
        Returns image size for image processing to be used in the model
        Args:
            model: Keras model instance
            Returns: img_size_model: Tuple of integers, image size
        """
        model_name = model.name
        if model_name == "vgg16":
            img_size_model = (224, 224)
        elif model_name == "resnet50":
            img_size_model = (224, 224)
        else:
            img_size_model = (224, 224)
        print(
            "Warning: model name unknown. Default image size: {}".format(img_size_model)
        )
        return img_size_model

    def get_layername_feature_extraction(self, model):
        """
        Return the name of last layer for feature extraction
        Args: model: Keras model instance
        Returns: layername_feature_extraction: String, name of the layer for feature extraction
        """
        model_name = model.name
        if model_name == "vgg16":
            layername_feature_extraction = "fc2"
        elif model_name == "resnet50":
            layername_feature_extraction = "predictions"
        else:
            layername_feature_extraction = ""
        print(
            "Warning: model name unknown. Default layername: '{}'".format(
                layername_feature_extraction
            )
        )
        return layername_feature_extraction

    def get_layers_list(self, model):
        """
        Get a list of layers from a model
        Args: model: Keras model instance
        Returns: layers_list: List of string of layername
        """
        layers_list = []
        for i in range(len(model.layers)):
            layer = model.layers[i]
            layers_list.append(layer.name)

        return layers_list

    def image_processing(self, img_array):
        """
        Preprocess image to be used in a keras model instance
        Args: img_array: Numpy array of an image which will be predicte
        Returns: processed_img = Numpy array which represents the processed image
        """
        # Expand the shape
        img = np.expand_dims(img_array, axis=0)

        # Convert image from RGB to BGR (each color channel is zero-centered with respect to the ImageNet dataset, without scaling)
        processed_img = preprocess_input(img)

        return processed_img

    def get_feature_vector(self, model, img_path):
        """
        Get a feature vector extraction from an image by using a keras model instance
        Args: model: Keras model instance used to do the classification.
                img_path: String to the image path which will be predicted
        Returns: feature_vect: List of visual feature from the input image
        """
        # Creation of a new keras model instance without the last layer
        layername_feature_extraction = self.get_layername_feature_extraction(model)
        model_feature_vect = Model(
            inputs=model.input,
            outputs=model.get_layer(layername_feature_extraction).output,
        )

        # Image processing
        img_size_model = self.get_img_size_model(model)
        img = image.load_img(img_path, target_size=img_size_model)
        img_arr = np.array(img)
        img_ = self.image_processing(img_arr)

        # Visual feature extraction
        feature_vect = model_feature_vect.predict(img_)

        return feature_vect

    def main_Cnn(self):
        # Step 1 load pre-trained model
        model = self.load_model(self.selected_model, include_top=True)
        target_path = ""
        feature_vect_list = []
        for i in range(0, len(self.image_list)):
            target_path = self.img_path + "" + self.image_list[i]
            feature_vect_list.append(self.get_feature_vector(model, target_path))

        return feature_vect_list
