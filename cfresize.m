function [im scale] = cfresize(im, longDimSize)
%CFRESIZE resize the image if its long dimension is bigger then longDimSize 
%   [im scale] = cfresize(im,longDimSize); scales the image so its long 
%   dimension is logDimSize long. (only if its bigger)   

s = size(im);
scale = min(longDimSize/max(s),1);
im = imresize(im, scale);