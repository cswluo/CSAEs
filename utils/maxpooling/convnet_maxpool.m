% convnet_maxpool
% Copyright (C) 2013 KyungHyun Cho
%
%This program is free software; you can redistribute it and/or
%modify it under the terms of the GNU General Public License
%as published by the Free Software Foundation; either version 2
%of the License, or (at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program; if not, write to the Free Software
%Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
%

% TODO: replace it with a more efficient routine.
function [out, outmap] = convnet_maxpool (in, ratio, varargin)

if nargin == 3
    stride = varargin{1};
else
    stride = ratio;
end

try
   use_gpu = gpuDeviceCount;
catch errgpu
   use_gpu = false;
   disp(['Could not use CUDA. Error: ' errgpu.identifier])
end


% because of different styles of indexing, some permutation is required.
% to use outidx, one needs to permute the original images in a batch and
% use the outidx as a linear index

if use_gpu
    outmap = gpuArray.zeros(size(in));
    [out, outidx] = MaxPooling(single(gather(in)), single([ratio ratio]));    
    outmap(outidx) = 1;    
    out = gpuArray(out);
else
    outmap = zeros(size(in));
    if stride == ratio % non-overlapping pooling
        [out, outidx] = MaxPooling(double(in), [ratio ratio]);
    else
        % overlapping pooling
        [out, outidx] = olapMaxPooling(double(in), [ratio ratio], [stride stride]);
        outidx = find(outidx);
    end
%     outmap = permute(outmap, [2, 3, 1]);
    
    outmap(outidx) = 1;
%     outmap = permute(outmap, [3, 1, 2]);
%     out = permute(out, [3, 1, 2]);
end

%w = size(in, 2);
%h = size(in, 3);

%wsub = floor(w / ratio);
%hsub = floor(h / ratio);

%if use_gpu
%    out = parallel.gpu.GPUArray.zeros(size(in, 1), wsub, hsub);
%    outidx = parallel.gpu.GPUArray.zeros(size(in, 1), wsub, hsub);
%else
%    out = zeros(size(in, 1), wsub, hsub);
%    outidx = zeros(size(in, 1), wsub, hsub);
%end


%for idx = 1:size(in, 1)
%    iin = reshape(in(idx,:,:), [w, h]);
%    iout = blockproc(iin, [ratio, ratio], @maxfilter);
%    ioutidx = blockproc(iin, [ratio, ratio], @maxidxfilter);
%%     for wi = 1:wsub
%%         for hi = 1:hsub
%%             %patch = squeeze(in(idx, ((wi-1)*ratio+1):(wi*ratio),((hi-1)*ratio+1):(hi * ratio)));
%%             %linpatch = patch(:);
%%             %linpatch = reshape(in(idx, ((wi-1)*ratio+1):(wi*ratio),((hi-1)*ratio+1):(hi * ratio)), [ratio*ratio,1]);
%%             if use_gpu
%%                 mask = parallel.gpu.GPUArray.ones(w,h);
%%             else
%%                 mask = ones(w,h);
%%             end
%%             mask(((wi-1)*ratio+1):(wi*ratio),((hi-1)*ratio+1):(hi * ratio)) = 0;
%%             mask = mask .* iin;
%%             linpatch = mask(:);
%%             [mv, midx] = max(linpatch);
%%             out(idx, wi, hi) = mv;
%%             outidx(idx, wi, hi) = midx;
%%             clear mask;
%%         end
%%     end
%    out(idx, :, :) = iout(1:wsub, 1:hsub);
%    outidx(idx, :, :) = ioutidx(1:wsub, 1:hsub);
%end

%end

%function [m] = maxfilter(block)

%    [m] = max(block.data(:));

%end
%function [idx] = maxidxfilter(block)

%    [idx] = max(block.data(:));

%end





