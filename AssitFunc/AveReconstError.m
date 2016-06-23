function aveError = AveReconstError(xtrain, poolstride, vneighbors, varargin)  

if nargin < 4
    error('Must specify the location of learned kernels.\n');
else
    kerlocations = varargin{1};
end

load(kerlocations);
numTrains = size(xtrain,2);
imgsize = repmat(sqrt(size(xtrain,1)),1,2);
if numTrains > 5000 
    index = randperm(numTrains, 5000);
    nloop = 5000;
else
    index = randperm(numTrains);
end

derror = [];
for loop = 1:nloop
    
    img = reshape(xtrain(:,index(loop)), imgsize);
    
    [~, rerror]  = ReconstError(img, 'max', kernels, hbias, obias, poolstride, vneighbors);
    
    
    %% error and display
    derror = [derror, rerror];
    
    
    %             figure(2);
    %             if strcmpi(datasetName,'CIFAR10')
    %                 show_centroids(reshape(kernels, size(kernels,1) * 3, [])' * 20, kernelsize1);
    %             else
    %                 display_network(kernels);  title 'Kernels'
    %             end
    %             figure(3); subplot(131), imshow(img,[]);  title 'Original image'%imagesc(img); title 'Original image'
    %             subplot(132); imshow(ri, []);  title 'Reconstructed image'%imagesc(img+E); title 'Reconstructed image'
    %             subplot(133); imshow(img - ri,[]);   title 'Residue image'%imagesc(-E); title 'Residue image'
    
    
end
aveError = mean(derror);
end

function  [rimg, error]  = ReconstError(X, acttype, dkernels, hbias,  obias, stride, nneighbors)

ekernels = flipud(dkernels);


Xsize       = size(X);
Ksize       = floor(sqrt(size(dkernels,1)));
Zsize       = Xsize(1) - Ksize + 1;
nhidmaps    = length(hbias);
ninputs     = size(X,3);

encoder     = zeros(Zsize, Zsize, nhidmaps);
decoder     = zeros(Xsize);



for i = 1:nhidmaps
    ek{i} = reshape(ekernels(:,(i-1)*ninputs+1:i*ninputs), Ksize, Ksize,ninputs);
    dk{i} = reshape(dkernels(:,(i-1)*ninputs+1:i*ninputs), Ksize, Ksize,ninputs);
end

%%% feed-forward
for i = 1:nhidmaps
    encoder(:,:,i) = convn(X, ek{i}, 'valid') + hbias(i);
end

switch acttype
    case 'max'
        mapp = feval(@maxAct, encoder);
    case 'sigmoid'
        mapp = feval(@sigmoidAct, encoder);
    case 'tanh'
        mapp = feval(@tanhAct, encoder);
    otherwise
        return;
end
[map, ind] = convnet_maxpool(mapp, stride);

%%% competition between neighbors
if nneighbors ~= nhidmaps
    rmapsize = size(map);
    [~, sind] = sort(map, 3, 'descend');
    sind = sind(:,:,1:nneighbors);
    sind = reshape(permute(sind,[3,1,2]),nneighbors,rmapsize(1)*rmapsize(2));
    indpool = zeros(rmapsize(3),rmapsize(1)*rmapsize(2));
    for i = 1:size(indpool,2)
        indpool(sind(:,i),i) = 1;
    end
    indpool = reshape(permute(indpool,[2,3,1]), size(map));
    map = map .* indpool;
end


%%% unpooling
if size(map,4) ~= 1
    map = expand(map, [stride stride 1 1]);
else
    map = expand(map, [stride stride 1]);
end
extSize = size(map);
if  extSize(1) > Zsize | extSize(2) > Zsize
    map(Zsize+1:end,:,:,:) = [];
    map(:,Zsize+1:end,:,:) = [];
end
map = map .* ind;


%%% decoder
for i = 1:ninputs
    tempout = zeros(Xsize(1),Xsize(2));
    for j = 1:nhidmaps
        tempout = tempout + convn(map(:,:,j), dk{j}(:,:,end-i+1), 'full');
    end
    decoder(:,:,i) = tempout;
end
rimg = decoder + obias;
error      =  0.5 * sum((rimg(:) - X(:)).^2);
end

