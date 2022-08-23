function varargout = Diabetic_Retinopathy_SVM(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Diabetic_Retinopathy_SVM_OpeningFcn, ...
                   'gui_OutputFcn',  @Diabetic_Retinopathy_SVM_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function Diabetic_Retinopathy_SVM_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
warning('off')

guidata(hObject, handles);


function varargout = Diabetic_Retinopathy_SVM_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function pushbutton1_Callback(hObject, eventdata, handles)


[filename,Filepath]=uigetfile('*.*','Select Test Image');   % Browse Image
filename=strcat(Filepath,filename);
Image = imread(filename); % Read Image
axes(handles.axes1)
imshow(Image);
handles.Image = Image;
guidata(hObject, handles);



% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)

h = msgbox('Processing');
load 'feature_texture.mat';
addpath 'CircleFinder';
pwd;                                                                        % Identify current folder
current_folder=pwd;
green = handles.Image(:,:,2);                                               % Select Green Channel of RGB Image
heq = adapthisteq(green);                                                   % Perform Adaptive Histogram Equalization
[pixelCountsG GLs] = imhist(green);                                         % Obtain the pixel counts

% Ignore 0
pixelCountsG(1) = 0;

% Find where histogram falls to 0.1% of the peak, on the bright side.
tIndex = find(pixelCountsG >= 0.001*max(pixelCountsG), 1, 'last');
thresholdValue = GLs(tIndex);                                               % Obtain the threshold value
binaryGreen = heq > thresholdValue;                                         % Obtain the binary image
binaryImage1 = bwareaopen(binaryGreen, 2000);                               % Remove blobs with size more than 2000 pixels to obtain optic disc
green1 = heq;
green1(~binaryImage1) = 0;                                                  % Segment the optic disc 
op = heq - green1;                                                          % Remove optic disc
law = laws(op,1);                                                           % Apply Law's Filter
law1 = double(law{1,1}); law2 = double(law{1,2}); law3 = double(law{1,3}); law4 = double(law{1,4});law5 = double(law{1,5}); law6 = double(law{1,6}); law7 = double(law{1,7}); law8 = double(law{1,8}); law9 = double(law{1,9});
featureSet = cat(3,law1,law2,law3,law4,law5,law6,law7,law8,law9);           % Concatenate all law's filter output
[m,n,z] = size(featureSet);

for i2 = 1:z
    g1 = graycomatrix(featureSet(:,:,i2));      % Obtain GLCM
    % Extract GLCM features (texture based features)
    [out] = GLCM_Features1(g1,0);   
    Autocorrelation = out.autoc;
    Contrast = out.contr;
    CorrelationM = out.corrm;
    CorrelationP = out.corrp;    
    Cluster_Prominence = out.cprom;
    Cluster_Shade = out.cshad;
    Dissimilarity = out.dissi;
    Energy = out.energ;
    Entropy = out.entro;
    HomogeneityM = out.homom;
    HomogeneityP = out.homop;
    Maximum_Probability = out.maxpr;
    Sum_of_Squares = out.sosvh;
    Sum_average = out.savgh;
    Sum_variance = out.svarh;
    Sum_entropy = out.senth;
    Difference_variance = out.dvarh;
    Difference_entropy = out.denth;
    Correlation1 = out.inf1h;
    Correlation2 = out.inf2h;
    Inv_Diff_Nor = out.indnc;
    Inv_Diff_M_Nor = out.idmnc;
    f = [Autocorrelation,Contrast,CorrelationM,CorrelationP,Cluster_Prominence,Cluster_Shade,Dissimilarity,Energy,Entropy,HomogeneityM,HomogeneityP,Maximum_Probability,Sum_of_Squares,Sum_average,Sum_variance,Sum_entropy,Difference_variance,Difference_entropy,Correlation1,Correlation2,Inv_Diff_Nor,Inv_Diff_M_Nor];
    f1(:,(22*i2-21):(22*i2)) = f;
end

test_feat = f1;

%% Classification Using SVM Classifier

l = [1;2];
label = [repmat(l(1),12,1);repmat(l(2),12,1)];
result = multisvm(feature_texture,label,test_feat);

if (result == 1)
    r = 'Patient is Healthy';
elseif (result == 2)
    r = 'Patient Suffers from Diabetic Retinopathy';
end

set(handles.edit1, 'string',r);

%% Extract Exudates 

if (result == 2)
    I = handles.Image;
    green = I(:,:,2);                                                       % Extracting green channel
    %histogram(green);
    [greenCounts, greenBinValues] = imhist(green);                          % Finding histogram of image data
    maxValueBin = find(greenCounts > 0, 1, 'last');                         % Considering only the max values (bigger regions)  
    minValueBin = find(greenCounts > 1200, 1, 'last');                      % Considering only the min values (smaller regions)
    bw = green > minValueBin & green <= maxValueBin;                        % Consists of only white portion 
    
    % For Removing OD

    hsv = rgb2hsv(I);
    Ihsv = hsv(:,:,3);
    im=adapthisteq(Ihsv);                                                   % Contrast-Limited Adaptive Histogram Equalization (CLAHE) algorithm was applied 
    [M,N] = size(im);
    % finds the circles
    [r c rad] = circlefinder(im);                                           % Finds Circle
    imageSizeX = M;
    imageSizeY = N;
    [columnsInImage rowsInImage] = meshgrid(1:imageSizeY, 1:imageSizeX);

    % Next create the circle in the image.
    centerX = c;
    centerY = r;
    radius = rad;
    circlePixels = (columnsInImage - centerX).^2 ...
    + (rowsInImage - centerY).^2 <= radius.^2;

    str = strel('disk',6,8);
    circlePixels = imdilate(circlePixels,str);                              % OD

    od = 1 - circlePixels;                                                  % Aleration of black and white pixels (for removing OD)
    op = bw .* od;                                                          % Contains only exudates 
    I1 =  heq + uint8(op);
    


%% Extract Microaneurysms & Hemorrhages
    I = handles.Image;
    red = I(:,:,1);
    hq = adapthisteq(red);                                                  % CLAHE
    hq = imadjust(hq);                                                      % Adjusting image intensity values
    hq = double(hq);
    
    % Alternating Sequential Filtering (ASF) 
    for iteration = 1 : 3
        radius = 2;                                                         
        r = radius * iteration;
        se = strel('disk',r,8);                                             % Step 1: using a disk shaped SE of variable radius from 2 pixels to 6 pixels
        imageArray = imopen(hq, se);
        imageArray = imclose(imageArray, se);
    end
    
    I1 = hq - imageArray;                                                   % Step 2: The resulting image of the CLAHE was subtracted from the resulting image of the ASF, obtaining an intensity inverted image
    a = imhmin(I1,30);                                                      % Regions of low contrast
    a1 = imhmin(a,1);                                                       % Regions of low contrast
    b = a1 - a;
    b1 = 1 - b;                                                             % Now, regions of low contrast, 
                                                                            % which could be associated with the optical disc, 
                                                                            % were eliminated using the H-minima transform   
  % Detecting Blood vessels
    % For the detection of the blood vessels, the ASF was again performed on the resulting image of the CLAHE, 
    % using now only one iteration. It was used a disc-shaped structuring element B with radius of 9 pixels
    iteration1 = 1; 
    radius1 = 9;
    r1 = radius1 * iteration1;
    se = strel('disk',r1,8);
    bw = imopen(hq, se);
    bw = imclose(bw, se);
    bw1 = bw - hq;                                                          % Enhancement of the low intensity structures
    bw3 = im2bw(bw1);                                                       
    
    nose = 12;    
    for k = 1 : nose
        th = 15 * k;
        se1 = strel('line',1,th);
        im_open = imopen(bw1,se1);
        img_open(:,:,k) = im_open;
    end
    % Finally, the 12 images obtained were added, getting the image img_open1, containing a sketch of the blood vessels
    img_open1 = img_open(:,:,1) + img_open(:,:,2) + img_open(:,:,3) + img_open(:,:,4) + img_open(:,:,5) + img_open(:,:,6) + img_open(:,:,7) + img_open(:,:,8) + img_open(:,:,9) + img_open(:,:,10) + img_open(:,:,11) + img_open(:,:,12);
    
    % The detection of the blood vessels
      img1 = imhmin(img_open1,60);
    img1 = im2bw(img1);
    img2 = bwmorph(img1,'dilate');
    img2 = bwmorph(img2,'open');
    bw2 = bwareaopen(img1,700);
    
    bb = bw3 - bw2;
    bw4 = bwlabel(bb,4);
    bw4 = bwareaopen(bw4,100);
    [bw4,num] = bwlabel(bw4,8);
    close(h)
    h = msgbox('Processed');
    pause(1)
    close(h)
    h = msgbox('Displaying Plots');
    pause(3)
    close(h);
    fig = figure(1);
    fig.Position =  [230 250 570 510];
    imshow(green),title('Green channel image');
    fig = figure(2);
    fig.Position =  [250 270 570 510];
    imshow(heq),title('Histogram equalised image');
    fig = figure(3),
    fig.Position =  [290 310 570 510];
    imshow(od),title('Optic disc segmention image');

    fig = figure(4),
    fig.Position =  [270 290 570 510];
    imshow(img2),title('Binary image');

    fig = figure(5),
    fig.Position =  [350 370 570 510];
    imshow(op,[]),title('Exudates with Optic disc eliminated image');

    fig = figure(6),
    fig.Position =  [370 390 570 510];
    imshow(bw4),title('Microaneurysms & Hemorrhages');
else
    close(h);
    h = msgbox('Processed');
    pause(2)
    try
    close(h)
    catch
    end
end
clc

function edit1_Callback(hObject, eventdata, handles)

function edit1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function pushbutton3_Callback(hObject, eventdata, handles)

load 'feature_texture12.mat';
pwd;
current_folder=pwd;

file = uigetdir('C:\Users\shahv\Desktop\project\T18 - 1658 Diabetic Retinopathy\');
imfile = dir(file);
imfile = imfile(3:end,1);

h = msgbox('Calculating....');      
for i = 1:length(imfile)
     Image = imread(fullfile(file,imfile(i).name));                         % Read Image
     green = Image(:,:,2);                                                  % Select Green Channel of RGB Image
     heq = adapthisteq(green);                                              % Perform Adaptive Histogram Equalization
     [pixelCountsG GLs] = imhist(green);                                    % Obtain the pixel counts
            
        % Ignore 0
        pixelCountsG(1) = 0;
        
        tIndex = find(pixelCountsG >= 0.001*max(pixelCountsG), 1, 'last');
        thresholdValue = GLs(tIndex);                                       % Obtain the threshold value
        binaryGreen = heq > thresholdValue;                                 % Obtain the binary image
        binaryImage1 = bwareaopen(binaryGreen, 2000);                       % Remove blobs with size more than 2000 pixels to obtain optic disc
        green1 = heq;
        green1(~binaryImage1) = 0;                                          % segment the optic disc 
        op = heq - green1;                                                  % Remove optic disc
        law = laws(op,1);                                                   % Apply Law's Filter
        law1 = double(law{1,1}); law2 = double(law{1,2}); law3 = double(law{1,3}); law4 = double(law{1,4});law5 = double(law{1,5}); law6 = double(law{1,6}); law7 = double(law{1,7}); law8 = double(law{1,8}); law9 = double(law{1,9});
        featureSet = cat(3,law1,law2,law3,law4,law5,law6,law7,law8,law9);   % Concatenate all law's filter output
        [m,n,z] = size(featureSet);
         
        for i2 = 1:z
            g1 = graycomatrix(featureSet(:,:,i2));                          % Obtain GLCM
            % Extract GLCM features (texture based features)
            [out] = GLCM_Features1(g1,0);
            Autocorrelation = out.autoc;
            Contrast = out.contr;
            CorrelationM = out.corrm;
            CorrelationP = out.corrp;    
            Cluster_Prominence = out.cprom;
            Cluster_Shade = out.cshad;
            Dissimilarity = out.dissi;
            Energy = out.energ;
            Entropy = out.entro;
            HomogeneityM = out.homom;
            HomogeneityP = out.homop;
            Maximum_Probability = out.maxpr;
            Sum_of_Squares = out.sosvh;
            Sum_average = out.savgh;
            Sum_variance = out.svarh;
            Sum_entropy = out.senth;
            Difference_variance = out.dvarh;
            Difference_entropy = out.denth;
            Correlation1 = out.inf1h;
            Correlation2 = out.inf2h;
            Inv_Diff_Nor = out.indnc;
            Inv_Diff_M_Nor = out.idmnc;
            f = [Autocorrelation,Contrast,CorrelationM,CorrelationP,Cluster_Prominence,Cluster_Shade,Dissimilarity,Energy,Entropy,HomogeneityM,HomogeneityP,Maximum_Probability,Sum_of_Squares,Sum_average,Sum_variance,Sum_entropy,Difference_variance,Difference_entropy,Correlation1,Correlation2,Inv_Diff_Nor,Inv_Diff_M_Nor];
            f1(:,(22*i2-21):(22*i2)) = f;
        end
        test_feature(i,:) = f1;
end
for k = 1:198
    test_feature1(:,k) = zscore(test_feature(:,k));
end

% Classification Using SVM Classifier

l = [1;2];
train_label = [repmat(l(1),40,1);repmat(l(2),40,1)];
test_label  =  [repmat(l(2),10,1);repmat(l(1),10,1)];

result = multisvm(feature_texture12,train_label,test_feature1);
CP_eu = classperf(test_label,result);
accuracy = CP_eu.CorrectRate * 100;
set(handles.edit2, 'string',accuracy);
close(h);
h = msgbox('Calculated');
pause(3);
close(h);


function edit2_Callback(hObject, eventdata, handles)

function edit2_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
