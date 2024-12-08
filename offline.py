from PIL import Image
from pathlib import Path
import numpy as np 

from feature_extractor import FeatureExtractor


if __name__=="__main__":
    fe=FeatureExtractor()
    # Loop through images and extract features
    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)
        # extract a deep feature here
        feature=fe.extract(img=Image.open(img_path))
        # Define the path to save the extracted feature
        feature_path=Path("./static/feature") / (img_path.stem + ".npy")
        print(feature_path)

        #save the feature
        np.save(feature_path,feature)