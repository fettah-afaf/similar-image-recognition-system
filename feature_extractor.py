from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.models import Model
import numpy as np 

class FeatureExtractor:
    def __init__(self):
        base_model=VGG16(weights="imagenet")
        self.model=Model(inputs=base_model.input,outputs=base_model.get_layer("fc1").output)
    def extract(self,img):
        img = img.resize((224,224)).convert("RGB")   
        x= image.img_to_array(img)  #to np.array
        x=preprocess_input(x) #substract avg pixelvalue
        x = np.expand_dims(x, axis=0)  # Add a batch dimension
        feature=self.model.predict(x)[0] # Extract features using the pre-trained model
        return feature / np.linalg.norm(feature)  #normalize the feature vector
