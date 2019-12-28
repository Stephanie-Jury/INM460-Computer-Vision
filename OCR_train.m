% Set up workspace
clc
clear all
close all
rng(1);

%% Access training data and create datastore

training_data_root = '/Users/steffjury/Desktop/Data Science MSc/Computer Vision/Coursework/Dataset/';
training_data_classes = fullfile(training_data_root, 'Numbers');
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

%% Create label vectors

training_labels_cat_SVM = categorical(training_imds.Labels);
validation_labels_cat_SVM = categorical(validation_imds.Labels);

%% Create HOG features

for i = 1:training_imds_size
    img = readimage(training_imds, i);
    training_feature_vector_HOG(i, :) = extractHOGFeatures(img, 'Cellsize', [8,8]);
end

for i = 1:validation_imds_size
    img = readimage(validation_imds, i);
    validation_feature_vector_HOG(i, :) = extractHOGFeatures(img, 'Cellsize', [8,8]);
end

%% SVM with HOG features (Accuracy: 98.93%)

% Train model
OCR_mdl = fitcecoc(training_feature_vector_HOG, training_labels_cat_SVM);
save('OCR_mdl.mat', 'OCR_mdl') % Save trained classifier as 'OCR_mdl'

% Load saved classifier
load OCR_mdl.mat

% Test model
OCR_predict = predict(OCR_mdl, validation_feature_vector_HOG);

% Plot confusion matrix
figure(2)
plotconfusion(validation_labels_cat_SVM, OCR_predict)
title('Confusion Matrix: SVM Classifier Trained on HOG Features', 'FontSize', 14)
ax = gca;
set(findobj(ax,'type','text'),'fontsize',3) 
set(gcf,'color','w');
hold off

% Output accuracy score
accuracy_SVM_HOG = sum(validation_labels_cat_SVM == OCR_predict)/length(validation_labels_cat_SVM)*100;
accuracy_round_SVM_HOG = round(accuracy_SVM_HOG,3);
strcat('SVM_HOG Classification Accuracy: ',  num2str(accuracy_round_SVM_HOG),'%')