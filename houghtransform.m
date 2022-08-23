function [houghTrans maxVec] = houghtransform(im, radVec)
%HOUGHTRANSFORM circle hough transform on a binary image im. 
%   HOUGHTRANSFORM(im, radVec) preforms hough transform for finding circles
%   with radiuses indicated by radVec on a binary image im.
% 
%   Return values:
%       houghTrans - a 3 dimensional matrix holding the hough transform   
%       maxVec - a vector with the same length as radVec indicating 
%            the likeiyhood of a circle in the corresponding radius. 
%
%   See also CIRCLEFINDER, EXTRACTCIRCLES.

% By Kobi Nistel.

eps = 0.3; %for (trying) fixing bias for small circles.
s = size(im);
houghTrans = zeros(s(1),s(2), length(radVec));
him = zeros(s(1)+2,s(2)+2);
hs = size(him);
for n = 1:length(radVec)
    rad = radVec(n);       
    delta = 2/rad;
    deg = 0:delta:2*pi;
    pX = cos(deg)*rad;
    pY = sin(deg)*rad;
    color = 1/(2*pi*rad+eps); % or just 1 with bias for large circles;
    for i=1:s(1)
        for j=1:s(2)            
            if(im(i,j)>0)
                him = him*0;
                vi = min(max(round(i + pX + 1),1),hs(1));
                vj = min(max(round(j + pY + 1),1),hs(2));
                index = (vi-1) + (vj-1)*hs(1) + 1;
                him(index) = color;                
                houghTrans(:,:,n) = houghTrans(:,:,n) +him(2:(end-1), 2:(end-1));
            end
            
        end
    end
end

maxVec = squeeze(max(max(houghTrans)));

