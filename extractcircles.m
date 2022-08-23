function [r c rad maxVec] = extractcircles(houghTrans, tresh , radVec, maxVec)
%EXTRACTCIRCLES extract circles from transformed image.
%   EXTRACTCIRCLES(houghTrans, tresh , radVec, maxVec)
%   returning a list of circles for hough transformed image. 
%
% Arguments: 
%           houghTrans - the transformed image.
%           tresh - threshold for the circle values in the range (0,1]
%           radVec - a vector of radiuses matching the indexs of the 3rd
%                    dimention of the transformed image.
%           maxVec - (optional) a vector with the same length as radVec 
%                    indicating the likelihood of a circle in the
%                    corresponding radius. 
% 
% Return values:
%            r - vector of row coordinates of the circles.
%            c - vector of column coordinates of the circles.
%            rad - vector of radiuses of the circles. 
%
%   See also CIRCLEFINDER, HOUGHTRANSFORM.

% By Kobi Nistel.

if(nargin<3)
    maxVec = squeeze(max(max(houghTrans)));
end

radIndexList =  nonmaxsup1d(maxVec, tresh);


if( ~isempty(radIndexList) )
    
    if(radIndexList(1)==1 && length(radIndexList)>1) %for less false positives
        radIndexList=radIndexList(2:end);
    end
    
    r = zeros(400,1);
    c = zeros(400,1);
    rad = zeros(400,1);
    
    cirCount = 0;
    for n = 1:length(radIndexList)
        radIndex = radIndexList(n);
        [y,x] = nonmaxsuppts(houghTrans(:,:,radIndex), 2,  tresh);
        num = length(y);
        r(cirCount+1:cirCount+num) = y;
        c(cirCount+1:cirCount+num) = x;
        rad(cirCount+1:cirCount+num) = radVec(radIndex);
        cirCount=cirCount+num;
    end   
    r = r(1:cirCount);
    c = c(1:cirCount);
    rad = rad(1:cirCount);    
else
   r = [];
   c = [];
   rad = [];
end