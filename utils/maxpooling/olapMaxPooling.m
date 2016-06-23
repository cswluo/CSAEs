function [out, outidx] = olapMaxPooling(x, ratio, stride)
[h, w, d, n] = size(x);
oh = length(1:stride(1):h);
ow = length(1:stride(2):w);
out = zeros(oh, ow, d, n);
outidx = zeros(size(x));

x = padarray(x,[ratio(1)-1, ratio(2)-1], 'post');
[ah, aw, ~, ~] = size(x);

coord = reshape(1:ah*aw, ah, aw);
codidx = im2col(coord,ratio,'sliding');

for i = 1:n
    for j = 1:d
        tidx    = zeros(ah,aw);
        temp    = squeeze(x(:,:,j,i));
        temp    = temp + rand(size(temp))*1e-12;
        temp    = temp(codidx);

        
        vmax    = max(temp,[],1);
        
        lgidx   = temp == repmat(vmax,prod(ratio),1);
        lgidx   = reshape(lgidx, prod(ratio), h, []);
        tlgidx  = zeros(size(lgidx));
        tlgidx(:,1:stride(1):end,1:stride(2):end) = lgidx(:,1:stride(1):end,1:stride(2):end);
        lgidx   = setdiff(unique(reshape(tlgidx,prod(ratio),[]) .* codidx),0);
        tidx(lgidx)  = 1;
        tidx(h+1:end,:) = [];
        tidx(:,w+1:end) = [];
        outidx(:,:,j,i) = tidx;
        
        vmax    = reshape(vmax, h, w);
        out(:,:,j,i) = vmax(1:stride(1):end, 1:stride(2):end);
        
    end
end
end