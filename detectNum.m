function [results] = detectNum(FileName)

% -------------------------------------------------------------------------------------------------
    % Output:
        % The function returns a matrix (results) when passed and RGB image (I).
        % Results will be a matrix of size N x 1, where N is the number of people detected in the image.
        % The column will contain the digits recognised in the image.

    % Arguments:
        % The function takes one arguments, FileName, which should be a
        % string path.
            
% -------------------------------------------------------------------------------------------------

% Runs if an image file selected
if FileName(end-3:end) == ".jpg" || FileName(end-4:end) == ".jpeg"
    fprintf('Checking number of people in image. \n')
    img = imread(FileName); % Read image
    img_size = size(img);
    
    % Detect faces in supplied image
    face_detect = vision.CascadeObjectDetector('MergeThreshold',10, 'MinSize', [20,20], 'ScaleFactor', 1.1);

    % Generate bounding boxes
    bbox_raw = step(face_detect, img);
    bbox = sortrows(bbox_raw, 1, 'ascend');
    
    num_faces = size(bbox, 1)
    
    if num_faces > 1
        % Function for splitting image into individual people
        img_store = splitImage(bbox, num_faces, img);
                
        for i = 1:length(img_store)
            fprintf('Extracting characters from image file. \n')
            singlePersonDetectNumIMAGE(img_store{i}) 
        end
            
    else
        fprintf('Extracting characters from image file. \n')
        singlePersonDetectNumIMAGE(img) 
    end
        
% Runs if a video file selected
elseif FileName(end-3:end) == ".mov"
    fprintf('Extracting characters from video file. \n')
    video = VideoReader(FileName); % Read video
    video_info = get(video);
    frames_num = video.NumberOfFrames; % Get number of frames
    frames_read = 1:5:frames_num; % Select every fifth frame
    detection_from_frame_store = [];
    bbox_from_frame_store = [];
    frame_show_store = [];
   
    % Loop over each selected frame
    for k=1:length(frames_read) 
         frame_idx = frames_read(k);
         current_frame  = read(video,frame_idx);
         
        % Check if there are multiple people in the image
        face_detect = vision.CascadeObjectDetector('MergeThreshold',10, 'MinSize', [20,20], 'ScaleFactor', 1.1);
        bbox_check = step(face_detect, current_frame);
        bbox_check = sortrows(bbox_check, 1, 'ascend');
        num_faces = size(bbox_check, 1);
        
        if num_faces > 1
            % Function for splitting image into individual people
            img_store = splitImage(bbox, num_faces, img);
            
            for i = 1:length(img_store) 
                try
                    % Detect individual from frame image 
                    [detection_from_frame, bbox_from_frame] = singlePersonDetectNumVIDEO(current_frame);
                    detection_from_frame_store = [detection_from_frame_store; detection_from_frame];
                    bbox_from_frame_store = [bbox_from_frame_store; bbox_from_frame];
                    frame_show_store = [frame_show_store; k];
                catch
                     % No detected characters to add
                end
            end
        else
             try
                 % Detect individual from frame image 
                 [detection_from_frame, bbox_from_frame] = singlePersonDetectNumVIDEO(current_frame);
                 detection_from_frame_store = [detection_from_frame_store; detection_from_frame];
                 bbox_from_frame_store = [bbox_from_frame_store; bbox_from_frame];
                 frame_show_store = [frame_show_store; k];
             catch
                 % No detected characters to add
             end
        end
    end
    
    % Find position of most commonly selected label
    try
        detection_from_frame_array = str2num(detection_from_frame_store);
    catch
        results = [];
        return
    end
    mode_label = mode(detection_from_frame_array);
    mode_bool = [detection_from_frame_array == mode_label];
    bbox_position_idx = bbox_from_frame_store(mode_bool,:);
    bbox_position = bbox_position_idx(1,:);
    frame_position_idx = frame_show_store(mode_bool);
    frame_show = read(video,frame_position_idx(1));
    figure(5);
    digit_show = insertObjectAnnotation(frame_show,'rectangle',bbox_position,mode_label,'FontSize',18);
    imshow(digit_show)
    results = mode_label;
               
else
    error('Error: Invalid file type supplied. Please supply a .jpg or .mov file.')
end

function [results] = singlePersonDetectNumIMAGE(img)
    figure
    imshow(img)
    
    % Create a mask over character area
    I = rgb2gray(img); 
    I = roicolor(I,10,180); % Select region of interest as white area
    %imshow(I)
    %se = strel('disk',5); % Create structuring element
    %I = imclose(I,se); % Remove noise using structuring element
    I = imcomplement(I); %change black to white and white to black
    %imshow(I)
    
    % Detect MSER regions
    [mserRegions, mserConnComp] = detectMSERFeatures(I, ... 
        'RegionAreaRange',[200 8000],'ThresholdDelta',80); %%Increased threshold so it detects less nonsense regions(e.g, some pattern on the floor that kinda looks like it could be a region)
    
    % Use regionprops to measure MSER properties
    mserStats = regionprops(mserConnComp, I, 'BoundingBox', 'Eccentricity', ...
        'Solidity', 'Extent', 'Euler', 'Image', 'PixelValues', 'PixelValues','MeanIntensity','MaxIntensity', 'MinIntensity'...
        ,'ConvexArea','ConvexHull','Extent','Extrema','Image','MajorAxisLength','MinorAxisLength','Perimeter','Orientation');
    
    % Compute the aspect ratio using bounding box data.
    bbox = vertcat(mserStats.BoundingBox); 
    N = size(bbox, 1);
    fprintf(' Function found %u potential digits. \n', N)
        if N == 0 %%when there are no digits detected
                results = []; %keep empty if there are no digits detected
                error('Error: \nNo digits detected in the selected image, please select another.')
        else
            digit_store_char = [];
            digit_store_num = [];
            bbox_store = [];

            for i=1:N % Extract MSER regions
                
                % Selecting characteristics that are common to digits
                if (mserStats(i).Perimeter >= 100 ...
                        && mserStats(i).MaxIntensity < 1 ...
                        && mserStats(i).ConvexArea >= 400 ...
                        && abs(mserStats(i).Orientation) >= 75 ...
                        && abs(mserStats(i).Orientation) <= 900)
                    
                    a = round(bbox(i, 1),0); %creating boundries for image to be cropped
                    b = round(bbox(i, 2),0);
                    c = a+round(bbox(i, 3),0);
                    d = b+round(bbox(i, 4),0);
                    digit_region = img(b:d, a:c, :);
                    %imshow(digit_region)
                    mserStats(i);
                    
                    if size(digit_region, 1) >= 50 %filtering out images that are too small
                        digit_region_resize = imresize(digit_region, [227 227]);
                        %imshow(digit_region_resize)
                        se = strel('disk',5);
                        digit_region_resize = imclose(digit_region_resize,se);
                        digit_region_resize = imdilate(digit_region_resize,strel('disk',3)); %increase visibility of the character
                        digit_region_resize = imcomplement(digit_region_resize);
                        %imshow(digit_region_resize)
                        image_contrast =  max(digit_region_resize(:)) - min(digit_region_resize(:));
                        
                        digit{i} = {digit_region_resize}; 
                        digit_img = cell2mat(digit{i});
                        regularExpr = '\d';
                        digit_ocr = ocr(digit_img,'TextLayout','Block');
                        
                        if ~isempty(digit_ocr.Text)
                            % Number 1 regularly mistaken for letters I and l.
                            % If detected, replace with number 1.
                            if digit_ocr.Text(1) == 'I' || digit_ocr.Text(1) == 'l'
                                digit_ocr_regex = '1';

                            % Number 0 regularly mistaken for letter O.
                            % If detected, replace with number 0.
                            elseif digit_ocr.Text(1) == 'O'
                                digit_ocr_regex = '0';

                            % Number 5 regularly mistaken for letter S.
                            % If detected, replace with number 5.
                            elseif digit_ocr.Text(1) == 'S'
                                digit_ocr_regex = '5';

                            % Otherwise match against the numeric regular
                            % expression
                            else
                                digit_ocr_regex = cell2mat(regexp(digit_ocr.Text,regularExpr,'match'));
                            end
                        end
                                       
                        if ~isempty(digit_ocr_regex)
                            digit_read_num = str2num(digit_ocr_regex);
                            digit_store_num = [digit_store_num; digit_read_num];
                        end
                        
                        bbox_store = [bbox_store; bbox(i, :)];
                        try
                            digit_store_char = [digit_store_char; digit_ocr_regex];
                        catch
                        end
                       
                    end    
                end
            end
        end
        
        if length(digit_store_num) > 1
            % Check if bounding boxes are close enough to be considered the same number
            for j = 1:length(digit_store_char)
                for m = (j+1):length(digit_store_char)
                    % If the left-most positions of consecutive bounding boxes are closer together than 1.5x the
                    % height of a single number, consider them to constitute one number
                    if abs(bbox_store(j, 1) -  bbox_store(m, 1)) < 150
                        digit_store_concat = [];

                        if bbox_store(j, 1) > bbox_store(m, 1)
                            digit_store_concat{j} = strcat(digit_store_char(m), digit_store_char(j));
                            digit_store_char(m, :) = [];

                        else
                            digit_store_concat{j} = strcat(digit_store_char(j), digit_store_char(m));
                            digit_store_char(m) = [];
                        end
                        bbox_store(j, 1) = min(bbox_store(j, 1), bbox_store(m, 1));
                        bbox_store(j, 2) = max(bbox_store(j, 2), bbox_store(m, 2));
                        bbox_store(j, 3) = bbox_store(j, 3) + bbox_store(m, 3);
                        bbox_store(j, 4) = max(bbox_store(j, 4),bbox_store(m, 4)) + abs(bbox_store(j, 4) - bbox_store(m, 4));
                        bbox_store(m, :) = [];
                    end
                end
            end
           
        else
            digit_store_concat = digit_store_char;
        end        
        
        digit_show = img;
        
        % Check if the ORC function has detected both numbers
        % simultaneously
        if length(digit_store_char) >= 2
            try
                digit_show = insertObjectAnnotation(digit_show,'rectangle',bbox_store(1,:),digit_store_char,'FontSize',20);
                imshow(digit_show)
                results = digit_store_char
            catch
                results = [];
                error('Error: No numbers detected.')
                return
            end
        else
            for k = 1:length(digit_store_char)
                digit_show = insertObjectAnnotation(digit_show,'rectangle',bbox_store(k,:),digit_store_concat(k),'FontSize',40);
                imshow(digit_show)
            end
            try class(digit_store_concat) == 'cell'
                results = cell2mat(digit_store_concat);
            catch
                results = [];
                error('Error: No numbers detected.')
            end
        end
end
function [results, bbox_results] = singlePersonDetectNumVIDEO(img)

    % Create a mask over character area
    I = rgb2gray(img); 
    I = roicolor(I,10,180); % Select region of interest as white area
    %se = strel('disk',5); % Create structuring element
    %I = imclose(I,se); % Remove noise using structuring element
    I = imcomplement(I); %change black to white and white to black
    
    % Detect MSER regions
    [mserRegions, mserConnComp] = detectMSERFeatures(I, ... 
        'RegionAreaRange',[200 8000],'ThresholdDelta',50); %%Increased threshold so
    
    % Use regionprops to measure MSER properties
    mserStats = regionprops(mserConnComp, I, 'BoundingBox', 'Eccentricity', ...
        'Solidity', 'Extent', 'Euler', 'Image', 'PixelValues', 'PixelValues','MeanIntensity','MaxIntensity', 'MinIntensity'...
        ,'ConvexArea','ConvexHull','Extent','Extrema','Image','MajorAxisLength','MinorAxisLength','Perimeter','Orientation');
    
    % Compute the aspect ratio using bounding box data.
    bbox = vertcat(mserStats.BoundingBox); 
    N = size(bbox, 1);
        if N == 0 %%when there are no digits detected
                results = []; %keep empty if there are no digits detected
                error('Error: \nNo digits detected in the selected image, please select another.')
        else
            digit_store_char = [];
            digit_store_num = [];
            bbox_store = [];
            for i=1:N % Extract MSER regions
                
                % Selecting characteristics that are common to digits
                if (mserStats(i).Perimeter >= 10 ...
                       && mserStats(i).MaxIntensity < 1 ...
                       && mserStats(i).ConvexArea >= 350 ...
                       && mserStats(i).ConvexArea <= 550 ...
                       && abs(mserStats(i).Orientation) >= 70 ...
                       && abs(mserStats(i).Orientation) <= 900)
                    
                    a = round(bbox(i, 1),0); %creating boundries for image to be cropped
                    b = round(bbox(i, 2),0);
                    c = a+round(bbox(i, 3),0);
                    d = b+round(bbox(i, 4),0);
                    digit_region = img(b:d, a:c, :);
                    %imshow(digit_region);
                    mserStats(i);
                    
                    if size(digit_region, 1) >= 15 %filtering out images that are too small
                        digit_region_resize = imresize(digit_region, [227 227]);
                        %imshow(digit_region_resize);
                        se = strel('disk',5);
                        digit_region_resize = imclose(digit_region_resize,se);
                        digit_region_resize = imdilate(digit_region_resize,strel('disk',3)); %increase visibility of the character
                        digit_region_resize = imcomplement(digit_region_resize);
                        %imshow(digit_region_resize);
                        image_contrast =  max(digit_region_resize(:)) - min(digit_region_resize(:));
                        
                        digit{i} = {digit_region_resize}; 
                        digit_img = cell2mat(digit{i});
                        regularExpr = '\d';
                        digit_ocr = ocr(digit_img,'TextLayout','Block');
                        
                        % Number 1 regularly mistaken for letters I and l.
                        % If detected, replace with number 1.
                        if digit_ocr.Text(1) == 'I' || digit_ocr.Text(1) == 'l'
                            digit_ocr_regex = '1';
                        
                        % Number 0 regularly mistaken for letter O.
                        % If detected, replace with number 0.
                        elseif digit_ocr.Text(1) == 'O'
                            digit_ocr_regex = '0';
                            
                        % Number 5 regularly mistaken for letter S.
                        % If detected, replace with number 5.
                        elseif digit_ocr.Text(1) == 'S'
                            digit_ocr_regex = '5';
                        
                        % Otherwise match against the numeric regular
                        % expression
                        else
                            digit_ocr_regex = cell2mat(regexp(digit_ocr.Text,regularExpr,'match'));
                        end
                                       
                        if ~isempty(digit_ocr_regex)
                            digit_read_num = str2num(digit_ocr_regex);
                            digit_store_num = [digit_store_num; digit_read_num];
                        end
                        
                        bbox_store = [bbox_store; bbox(i, :)];
                        digit_store_char = [digit_store_char; digit_ocr_regex];
                       
                    end    
                end
            end
        end
        
        if length(digit_store_num) > 1
            % Check if bounding boxes are close enough to be considered the same number
            for j = 1:length(digit_store_char)
                for m = (j+1):length(digit_store_char)
                    % If the left-most positions of consecutive bounding boxes are closer together than 1.5x the
                    % height of a single number, consider them to constitute one number
                    if abs(bbox_store(j, 1) -  bbox_store(m, 1)) < 150
                        digit_store_concat = [];

                        if bbox_store(j, 1) > bbox_store(m, 1)
                            digit_store_concat{j} = strcat(digit_store_char(m), digit_store_char(j));
                            digit_store_char(m, :) = [];

                        else
                            digit_store_concat{j} = strcat(digit_store_char(j), digit_store_char(m));
                            digit_store_char(m) = [];
                        end
                        bbox_store(j, 1) = min(bbox_store(j, 1), bbox_store(m, 1));
                        bbox_store(j, 2) = max(bbox_store(j, 2), bbox_store(m, 2));
                        bbox_store(j, 3) = bbox_store(j, 3) + bbox_store(m, 3);
                        bbox_store(j, 4) = max(bbox_store(j, 4),bbox_store(m, 4)) + abs(bbox_store(j, 4) - bbox_store(m, 4));
                        bbox_store(m, :) = [];
                    end
                end
            end
           
        else
            digit_store_concat = digit_store_char;
        end        
        
        if length(digit_store_char) > 2
            results = digit_store_char;
            
        else
            if class(digit_store_concat) == 'cell'
                results = cell2mat(digit_store_concat);
            else
                results = cell2mat(digit_store_concat);
            end
        end
        
        bbox_results = bbox_store;
end
function [img_store] = splitImage(bbox, num_faces, img)
            img_store_temp = cell(num_faces, 1);
            left_position = bbox(1,1) + bbox(1,2);
            right_position = bbox(2,1);
            vertical_cut = (left_position + right_position)/2;
            img_1 = imcrop(img, [0, 0, vertical_cut, img_size(1)]);
            img_store_temp{1} = img_1;
            vertical_cut_store = vertical_cut;

            for n = 2:(num_faces-1)
                vertical_cut_store = [];
                left_position_1 = bbox(n,1) + bbox(n,2);
                right_position_1 = bbox(n+1,1);
                vertical_cut_1 = (left_position_1 + right_position_1)/2;
                vertical_cut_store = [vertical_cut_store; vertical_cut_1];
                left_position_2 = bbox(n+1,1) + bbox(n+1,2);
                if n == (num_faces-1)
                    final_cut = img_size(2)-vertical_cut_store(end);
                else
                    right_position_2 = bbox(n+2,1);
                    vertical_cut_2 = (left_position_2 + right_position_2)/2;
                    final_cut = vertical_cut_2 - vertical_cut_1;
                end

                img_n = imcrop(img, [vertical_cut, 0, final_cut, img_size(1)]);
                img_store_temp{n} = img_n;
            end

            img_end = imcrop(img, [vertical_cut_store(end), 0, (img_size(2)-vertical_cut_store(end)), img_size(1)]);
            img_store_temp{num_faces} = img_end;
            img_store = img_store_temp;
        end

end
