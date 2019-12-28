function [P] = RecogniseFace(I, featureType, classifierName)
    
% -------------------------------------------------------------------------------------------------
    % Ensure this function in the same folder as the five trained
    % classifier files and SURF_bag.mat

    % Output:
        % The function returns a matrix (P) when passed and RGB image (I).
        % P will be a matrix of size N x 3, where N is the number of people detected in the image.
        % The three columns represent:
            % 1 = ID: the unique number assigned to each individual during training
            % (ID will return 0 if the classifier is unable to identify the individual)
            % 2 = x: the x location of the central position of the detected face
            % 3 = y: the y location of the central position of the detected face

    % Arguments:
        % The function takes three arguments: 
            % I: image name
            % classifierName:  "SVM", "MLP", "AlexNet"
            % featureType: "HOG", "SURF", "None"
            % (NOTE: HOG or SURF must be specified when using SVM and MLP classifiers,
            % AlexNet takes "None" as the feature argument)
            
% -------------------------------------------------------------------------------------------------
    
    % Definte known class labels
    class_labels = ["01";"02";"03";"04";"05";"06";"07";"08";"09";"10";"11";...
        "12";"13";"14";"15";"16";"17";"20";"21";"22";"24";"33";"34";"36";"37";...
        "38";"39";"40";"41";"42";"43";"44";"45";"46";"47";"48";"49";"50";"51";...
        "52";"53";"54";"55";"56";"57";"58";"59";"60";"61";"62";"63";"64";"65";...
        "66";"67";"68";"69";"70";"71";"72";"73";"74";"75";"76";"77";"78";"79";...
        "80";"81";"82";"83"];

    % Read supplied image
    img = imread(I);
    
    % Show supplied image
    figure(1)
    imshow(img)

    % Detect faces in supplied image
    face_detect = vision.CascadeObjectDetector('MergeThreshold',10, 'MinSize', [20,20], 'ScaleFactor', 1.1);

    % Generate bounding boxes
    bbox = step(face_detect, img);
    num_faces = size(bbox, 1);

    % Initialise function output
    P = [];

    % If there are no faces detected do not update P, and display an error
    if num_faces == 0
        error('Error: \nNo faces detected in the supplied image.', I)

    % Otherwise, print the number of faces detected
    else  
        fprintf('\n%u faces deteced in the supplied image. \n', num_faces)
    end

    % Extract each face 
    for i=1:num_faces
        a = bbox(i, 1);
        b = bbox(i, 2);
        c = a+bbox(i, 3);
        d = b+bbox(i, 4);
        face = img(b:d, a:c, :); 
        
        % Resize each face to match those used in model training and store 
        face = imresize(face, [227 227]); 
        face_store{i} = {face}; 
        img_current_face = cell2mat(face_store{i});
        
        % Record the x and y coordinates of the face centre
        x = (a + c)/2;
        y = (b + d)/2;
        
        % Carry out prediction based on supplied feature extraction and classifier arguments
        
        % If feature extraction method selected is SURF
        if isequal (featureType, "SURF")
            
            % Extract SURF features
            load 'SURF_bag.mat'
            feature_vector_SURF = encode(SURF_bag, face);
       
            if isequal (classifierName, "SVM")
                % Load trained classifier 
                load SVM_SURF_mdl.mat;
                 
                % Predict label using SVM
                predicted_label_idx = grp2idx(predict(SVM_SURF_mdl, feature_vector_SURF));
                predicted_label = str2num(class_labels(predicted_label_idx));
                
            elseif isequal (classifierName, "MLP")
                % Load trained classifier 
                load MLP_SURF_mdl.mat;
                
                % Predict label using MLP
                predicted_label = MLP_SURF_mdl(feature_vector_SURF');
                [~, predicted_label_idx] = max(predicted_label(:,1)); 
                predicted_label = str2num(class_labels(predicted_label_idx));
             
            else
                error('Error: \nIncorrect classifier name selected, please choose another.', classifierName)
                x = 0; 
                y = 0;
                predicted_label = 0;
            end
       
        % If feature extraction method selected is HOG
        elseif isequal (featureType, "HOG") 
            
            % Extract HOG features
            HOG_features = extractHOGFeatures(img_current_face);
            
            if isequal (classifierName, "SVM")
                % Load trained classifier 
                load SVM_HOG_mdl.mat;
                
                % Predict label using SVM
                predicted_label = predict(SVM_HOG_mdl, HOG_features); 
                predicted_label = double(string(predicted_label));
            
            elseif isequal (classifierName, "MLP")
                % Load trained classifier 
                load MLP_HOG_mdl.mat
                
                % Predict label using MLP
                predicted_label_raw = MLP_HOG_mdl(HOG_features');
                [~, predicted_label_idx] = max(predicted_label_raw(:,1));
                predicted_label = str2num(class_labels(predicted_label_idx));
                
            else
                error('Error: \nIncorrect classifier name selected, please choose another.', classifierName)
                x = 0; 
                y = 0;
                predicted_label = 0;
            end
            
        elseif isequal (featureType, "None") && isequal (classifierName, "AlexNet")
            % Load trained classifier   
            load AlexNet.mat;
            
            % Predict label using AlexNet
            predicted_label_idx = grp2idx(classify(CNN_mdl,img_current_face));
            predicted_label = str2num(class_labels(predicted_label_idx));

        % If incorrect sequence of feature and classifier selected, print a warning
        else
            error('Error: \nIncorrect feature type selected, please choose another.', featureType)
            x = 0; 
            y = 0;
            predicted_label = 0;
        end
        
        % If no class found, set predicted label to zero
        if isempty(predicted_label)
            predicted_label = 0; 
        end    
        
        % Update P with the face location and predicted label
        P = [P; round(predicted_label,0), round(x,0), round(y,0)];
    end
    
    % Show image with labels
    figure(2)
    face_annotation = insertObjectAnnotation(img,'rectangle',bbox,P(:,1));   
    imshow(face_annotation)
    title('Detected faces');
    hold on
    
end