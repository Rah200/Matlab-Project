clc
close all
clear all

filename = 'D:\desktop-Rachita\DONE\Others\B-FGMN Diabetic Retinopathy\code\Test3\2h.tif';
I = imread(filename);

hsv = rgb2hsv(I);
Ihsv = hsv(:,:,3);
im=adapthisteq(Ihsv);
[M,N] = size(im);

% finds the circles
[r c rad] = circlefinder(im);


imageSizeX = M;
imageSizeY = N;
[columnsInImage rowsInImage] = meshgrid(1:imageSizeY, 1:imageSizeX);

% Next create the circle in the image.
centerX = c;
centerY = r;
radius = rad;

circlePixels = (columnsInImage - centerX).^2 ...
    + (rowsInImage - centerY).^2 <= radius.^2;

cdisk = im.*circlePixels;

figure(1)
imshow(circlePixels) ;
figure(2)
imshow(im);
figure(3)
imshow(cdisk);
