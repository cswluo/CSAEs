function pim = resize_im(img,sz)

% input has to be grayscale image
% resize the longer side of input image to sz

szim = size(img);
[maxs,maxi] = max(szim);
szn = [NaN NaN];

szn(maxi) = sz;
imres = imresize(img,szn,'bicubic');

%%% pad with zero
shift = floor(( sz - size(imres)) / 2);
pim = zeros(sz);
pim(1+shift(1):shift(1)+size(imres,1),1+shift(2):shift(2)+size(imres,2)) = imres;
end