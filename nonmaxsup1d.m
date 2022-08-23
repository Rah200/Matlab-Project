function [xList maxX] = nonmaxsup1d(x, thresh)
% finds the maxima points in x with value > thresh
% xList - indexs of the points;
% maxX - a vector with ones on the maxima enterys and zeros else
if(~isempty(x))
    x = [x(1);x;x(end)];
    dx = x(2:end)-x(1:end-1);
    zc = max(sign(sign(dx(1:end-1))-sign(dx(2:end))),0);
    tx = (x>thresh);
    maxX = tx(2:end-1).*zc;
    n = 1:length(maxX);
    xList =  n(maxX==1);
else
    xList = [];
    maxX = [];
end