Submission Contents:

1. training_set_augment.m 
    - The code used to augment the training data

2. train_val.m
    - The code used to train and test the classifiers for RecogniseFace.m

3. Five trained classifiers, these must be placed in the same folder as RecogniseFace.m
    - SVM_SURF_mdl.mat 
    - SVM_HOG_mdl.mat 
    - MLP_SURF_mdl.mat 
    - MLP_HOG_mdl.mat 
    - AlexNet.mat 

4. SURF_bag.mat
    - Bag of encoded SURF features for the training data, this must be placed in the same folder as RecogniseFace.m

5. RecogniseFace.m
    - This contains the RecogniseFace function

6. detectNum.m
    - This contains the detectNum function

7. "Test Images" folder
    - A selection of images and videos which can be supplied to RecogniseFace and detectNum to illustrate correct working of the functions

8. OCR_train.m
    - The code used to train a digit recognition model, but was not included in either function. 
