clc;
clear all;
close all;

%% 1. Generate face images from video files and group images

% Root Directory
root_path = '/Users/steffjury/Desktop/Data Science MSc/Computer Vision/Coursework/Dataset';
images_path = fullfile(root_path,'Training Images Faces');

% Get files and directories
files = dir(images_path);
directories = [files.isdir];

% Extract subdirectories
subdirectories = files(directories);

% Define the face detection method
face_detect = vision.CascadeObjectDetector('MergeThreshold',10, 'MinSize', [20,20]);

% Loop over subdirectories and extract faces from videos
for i = 3:length(subdirectories)
    
    % Get subfolders under original image folder
    subfolder = fullfile(images_path,subdirectories(i).name);
    subdirectories(i).name

    % Get all jpg files in folder
    image_files = dir(fullfile(subfolder,'*.jpg'));
    image_files
    
    % Get subfolders under original image folder
    subfolder = fullfile(images_path,subdirectories(i).name)

    % Get all mov files in folder
    files = dir(fullfile(subfolder));
    
    for j = 4:length(files)
        % Run face detection and save output

        image = imread(fullfile(fullfile(subfolder),files(j).name));   

        %Detect face
        bounding_box = step(face_detect,image);

        %Crop and resize faces
        for k = 1:size(bounding_box,1)
            face_crop = imcrop(image,bounding_box(k,:));
            face_227 = imresize(face_crop, [227,227]);
            name = strcat(subdirectories(i).name,'_',int2str(j),'_',int2str(k),'.jpg'); 
            imwrite(face_227, fullfile(subfolder, name))
            
        end
    end
end
        

%% 2. Train preliminary classifier on labelled individual images and apply to unlabelled faces extracted from group images 

% Access training data and create datastore
training_data_root = '/Users/steffjury/Desktop/Data Science MSc/Computer Vision/Coursework/Dataset/';
training_data_classes = fullfile(training_data_root, 'Training Images Faces');
imds = imageDatastore(training_data_classes, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Inspect label counts
counts = countEachLabel(imds);
min_count = min(counts.Count);
imds_equal = splitEachLabel(imds, min_count, 'randomized');
imds_equal_counts = countEachLabel(imds_equal);

% Get class names from folder names
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

% Use all labelled data for training
training_imds = imds_equal;
training_imds_size = numel(training_imds.Files);

% Create label vector for training
training_labels_cat_SVM = categorical(training_imds.Labels);

% Create SURF features
bag = bagOfFeatures(training_imds); 

% Encode SURF feature vectors
training_feature_vector_SURF = encode(SURF_bag, training_imds);

% Train model
SVM_SURF_mdl = fitcecoc(training_feature_vector_SURF, training_labels_cat_SVM);

% Use model to append predicted class name to input image files
image_files = dir(fullfile(subfolder,'*.jpg')); % Get all jpg files in folder

for j = 4:length(image_files)
    img = imread(image_files);
    test_feature_vector_SURF = encode(SURF_bag, img);
    SVM_SURF_predict = predict(SVM_SURF_mdl, test_feature_vector_SURF);
    name = strcat(int2str(SVM_SURF_predict),'_',int2str(j),'.jpg'); 
    imwrite(image_files(j), fullfile(subfolder, name));
end



