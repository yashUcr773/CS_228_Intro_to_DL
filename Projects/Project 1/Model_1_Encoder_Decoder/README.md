This folder contains the files required to train the Image colorization model and run inference to generated coloured images.

- The file `model_1_encoder_decoder_train.py` contains the code to 
    - download the dataset
    - create dataloaders
    - apply augmentation
    - create a model and train it.

- The file `model_1_encoder_decoder_inference.py` contains the code to 
    - use and existing model weights to generate colored images from dataset.