%% train_val.mat is trained on the 83 image classes

% Set up workspace
clc
clear all
close all
rng(1);

%% Access training data and create datastore

training_data_root = '/Users/steffjury/Desktop/Data Science MSc/Computer Vision/Coursework/Dataset/';
training_data_classes = fullfile(training_data_root, 'Training Images Faces');
imds = imageDatastore(training_data_classes,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Inspect label counts

counts = countEachLabel(imds);
min_count = min(counts.Count);
imds_equal = splitEachLabel(imds, min_count, 'randomized');
imds_equal_counts = countEachLabel(imds_equal);

%% Store list of class label names

% Get folder names
folder_info = struct2dataset(dir(training_data_classes));
folder_names_dataset = dataset2cell(folder_info(4:end,1));
folder_names = string(folder_names_dataset(2:end));

num_classes = length(folder_names);

class_labels = [];
class_labels_num = [];

for i = 1:num_classes
    imgSets(i) = imageSet(convertStringsToChars(fullfile(training_data_classes,folder_names(i))));
    class_labels = [class_labels; folder_names(i)];
    class_labels_num = [class_labels_num; str2num(folder_names(i))];
end


%% Split data into training and validation sets

[training_imds, validation_imds] = splitEachLabel(imds_equal, 0.7, 'randomized');

training_imds_size = numel(training_imds.Files);
validation_imds_size = numel(validation_imds.Files);

%% Create label vectors for use with each classifier

% SVM
training_labels_cat_SVM = categorical(training_imds.Labels);
validation_labels_cat_SVM = categorical(validation_imds.Labels);

% MLP (one-hot encoding)
training_num = categories(training_labels_cat_SVM);
validation_num = categories(validation_labels_cat_SVM);

training_labels_cat_MLP = zeros(numel(training_num), training_imds_size);
validation_labels_cat_MLP = zeros(numel(validation_num), validation_imds_size);

for i = 1:training_imds_size
    label = training_labels_cat_SVM(i);
    training_labels_cat_MLP(strcmp(training_num, cellstr(label)),i) = 1;
end

for i = 1:validation_imds_size
    label = validation_labels_cat_SVM(i);
    validation_labels_cat_MLP(strcmp(validation_num, cellstr(label)),i) = 1;
end

%% Create SURF features

%bag = bagOfFeatures(training_imds); 
%save('SURF_bag.mat', 'bag') % Save visual vocabulary as 'SURF_bag'

% Load saved visual vocabulary
load '/Users/steffjury/Desktop/Data Science MSc/Computer Vision/Coursework/SURF_bag.mat'

% Encode SURF feature vectors
training_feature_vector_SURF = encode(SURF_bag, training_imds);
validation_feature_vector_SURF = encode(SURF_bag, validation_imds);
 
%% Create HOG features

for i = 1:training_imds_size
    img = readimage(training_imds, i);
    training_feature_vector_HOG(i, :) = extractHOGFeatures(img, 'Cellsize', [8,8]);
end

for i = 1:validation_imds_size
    img = readimage(validation_imds, i);
    validation_feature_vector_HOG(i, :) = extractHOGFeatures(img, 'Cellsize', [8,8]);
end

%% 1. SVM with SURF features (Accuracy: 98.52%)

% Train and save model
%SVM_SURF_mdl = fitcecoc(training_feature_vector_SURF, training_labels_cat_SVM);
%save('SVM_SURF_mdl.mat', 'SVM_SURF_mdl') % Save trained classifier as 'SURF_SVM'

% Load saved classifier
load SVM_SURF_mdl.mat

% Test model
SVM_SURF_predict = predict(SVM_SURF_mdl, validation_feature_vector_SURF);

% Plot confusion matrix
figure(1)
plotconfusion(validation_labels_cat_SVM, SVM_SURF_predict)
title('Confusion Matrix: SVM Classifier Trained on SURF Features', 'FontSize', 12)
ax = gca;
set(findobj(ax,'type','text'),'fontsize',3) 
set(gcf,'color','w');
hold off

% Output accuracy score
accuracy_SVM_SURF = sum(validation_labels_cat_SVM == SVM_SURF_predict)/length(validation_labels_cat_SVM)*100;
accuracy_round_SVM_SURF = round(accuracy_SVM_SURF,3);
strcat('SVM_SURF Classification Accuracy: ',  num2str(accuracy_round_SVM_SURF),'%')

%% 2. SVM with HOG features (Accuracy: 98.93%)

% Train model
%SVM_HOG_mdl = fitcecoc(training_feature_vector_HOG, training_labels_cat_SVM);
%save('SVM_HOG_mdl.mat', 'SVM_HOG_mdl') % Save trained classifier as 'SURF_SVM'

% Load saved classifier
load SVM_HOG_mdl.mat

% Test model
SVM_HOG_predict = predict(SVM_HOG_mdl, validation_feature_vector_HOG);

% Plot confusion matrix
figure(2)
plotconfusion(validation_labels_cat_SVM, SVM_HOG_predict)
title('Confusion Matrix: SVM Classifier Trained on HOG Features', 'FontSize', 14)
ax = gca;
set(findobj(ax,'type','text'),'fontsize',3) 
set(gcf,'color','w');
hold off

% Output accuracy score
accuracy_SVM_HOG = sum(validation_labels_cat_SVM == SVM_HOG_predict)/length(validation_labels_cat_SVM)*100;
accuracy_round_SVM_HOG = round(accuracy_SVM_HOG,3);
strcat('SVM_HOG Classification Accuracy: ',  num2str(accuracy_round_SVM_HOG),'%')

%% 3. MLP with SURF features (Accuracy: 86.23%)

% Train model
%MLP_SURF_mdl = patternnet(50);
%MLP_SURF_mdl = train(MLP_SURF_mdl, training_feature_vector_SURF', training_labels_cat_MLP);
%nntraintool
%save('MLP_SURF_mdl.mat', 'MLP_SURF_mdl') % Save trained classifier as 'MLP_SURF'

% Load saved classifier
load MLP_SURF_mdl.mat

% Test model
MLP_SURF_predict_raw = MLP_SURF_mdl(validation_feature_vector_SURF');

% Convert predictions to vector
%validation_num_cat = categories(validation_num);
MLP_SURF_predict = [];

for i = 1:validation_imds_size
    [value, index] = max(MLP_SURF_predict_raw(:,i));
    MLP_SURF_predict = [MLP_SURF_predict; validation_num(index)];
end

MLP_SURF_predict = categorical(str2num(cell2mat(MLP_SURF_predict)));

% Plot confusion matrix
figure(3)
plotconfusion(validation_labels_cat_SVM, MLP_SURF_predict)
title('Confusion Matrix: MLP Classifier Trained on SURF Features', 'FontSize', 14)
ax = gca;
set(findobj(ax,'type','text'),'fontsize',3) 
set(gcf,'color','w');
hold off

% Output accuracy score
accuracy_MLP_SURF = sum(validation_labels_cat_SVM == MLP_SURF_predict)/length(validation_labels_cat_SVM)*100;
accuracy_round_MLP_SURF = round(accuracy_MLP_SURF,3);
strcat('MLP_SURF Classification Accuracy: ',  num2str(accuracy_round_MLP_SURF),'%')

%% 4. MLP with HOG features (Accuracy: 86.10%)

% Train model
MLP_HOG_mdl = patternnet(50);
MLP_HOG_mdl = train(MLP_HOG_mdl, training_feature_vector_HOG', training_labels_cat_MLP);
nntraintool
%save('MLP_HOG_mdl.mat', 'MLP_HOG_mdl') % Save trained classifier as 'MLP_SURF'

% Load saved classifier
load MLP_HOG_mdl.mat

% Test model
MLP_HOG_predict_raw = MLP_HOG_mdl(validation_feature_vector_HOG');

% Convert predictions to vector
MLP_HOG_predict = [];

for i = 1:validation_imds_size
    [value, index] = max(MLP_HOG_predict_raw(:,i));
    MLP_HOG_predict = [MLP_HOG_predict; validation_num(index)];
end

MLP_HOG_predict = categorical(str2num(cell2mat(MLP_HOG_predict)));

% Plot confusion matrix
figure(4)
plotconfusion(validation_labels_cat_SVM, MLP_HOG_predict)
title('Confusion Matrix: MLP Classifier Trained on HOG Features', 'FontSize', 14)
ax = gca;
set(findobj(ax,'type','text'),'fontsize',3) 
set(gcf,'color','w');
hold off

% Output accuracy score
accuracy_MLP_HOG = sum(validation_labels_cat_SVM == MLP_HOG_predict)/length(validation_labels_cat_SVM)*100;
accuracy_round_MLP_HOG = round(accuracy_MLP_HOG,3);
strcat('MLP_HOG Classification Accuracy: ',  num2str(accuracy_round_MLP_HOG),'%')

%% 5. Pretrained AlexNet CNN (Accuracy: 97.89%)

% Train model
CNN_mdl = alexnet;
layersTransfer = CNN_mdl.Layers(1:end-3);

layers = [
    layersTransfer
    fullyConnectedLayer(num_classes,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 50, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 0.0001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 1, ...
    'LearnRateDropFactor', 0.1, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationData', validation_imds, ...
    'ValidationFrequency', 50);

% CNN_mdl = trainNetwork(training_imds, layers, options);

% save('AlexNet.mat', 'CNN_mdl') % Save trained classifier

% Load saved classifier
load AlexNet.mat

% Test model
CNN_predict = classify(CNN_mdl, validation_imds);

% Plot confusion matrix
figure(5)
plotconfusion(validation_labels_cat_SVM, CNN_predict)
title('Confusion Matrix: AlexNet Classifier', 'FontSize', 14)
ax = gca;
set(findobj(ax,'type','text'),'fontsize',3) 
set(gcf,'color','w');
hold off

% Output accuracy score
accuracy_CNN = sum(validation_labels_cat_SVM == CNN_predict)/length(validation_labels_cat_SVM)*100;
accuracy_round_CNN = round(accuracy_CNN,3);
strcat('AlexNet Classification Accuracy: ',  num2str(accuracy_round_CNN),'%')
