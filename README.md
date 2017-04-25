"# Traffic-Sign-Classifier" 


The purpose of this program was to use tensor flow in order to classify street signs from the German Traffic Sign Data Base (GTSDB).

Generaly the program uses a LeNet network architecture with an Adam Optimizer in order to classify the images. 

The images undergo preprocessing to ensure that they are 32x32 pixels large, grey scaled, and normalized.

Please refer to the writeup.md for more detailed information on how this was accomplished.

#NOTE - The images are classified, however they are not identified. This program will determine what the most likely traffic sign in an image is, it will not tell you whether there is a traffic sign or not in an image (though with some adaption to the image size input and a large database of training data that could be possible).

