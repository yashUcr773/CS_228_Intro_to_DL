This folder contains the implementation for the project Image Colorization for CS 228 Intro to Deep Learning

- Run the file `scrapper.py` to scrape the unsplash website and download the images. 
- Images are set in some predefined categories and some 20 images per category are downloaded.
- Optional Step. View the images and screen them if they are not up to standard.

- Run the `dataset_creation.py` to convert the downloaded images to dataset by resizing them to 512x512 pixels and converting the coloured images to black_and_white.
- The images are stored in respective folders `black_and_white` and `true_labels`.
- The dataset can be found at [here](https://d1u36hdvoy9y69.cloudfront.net/cs-228-intro-to-dl/Project/dataset.zip)


The folder Model_1_Encoder_Decoder contains the code to train the model, run inference on already trained model and a readme file with information.        
