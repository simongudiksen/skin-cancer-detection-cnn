# skin-cancer-detection-cnn
CNN image classifier to detect skin cancer

In this script I use the pretrained image classifier "resnet50" to detect one of three classes: 'melanoma', 'nevus', 'seborrheic_keratosis'. I freeze every weight in the CNN part of the network. Then replace the last layer in the fully connected part and change the output number of neurons to 3 (number of desired classes)

I only train the fully connected layer. 
