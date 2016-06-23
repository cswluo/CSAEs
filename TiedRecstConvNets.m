function [dkernels, hbias, obias, params, decoder, rerror] = TiedRecstConvNets(X, acttype, dkernels,...
    hbias, obias, params, stride, varargin)

if nargin == 9
    tiedflag        = varargin{1};
    nneighbors      = varargin{2};
elseif nargin == 8
    tiedflag        = varargin{1};
    nneighbors      = 5;
else
    tiedflag        = 1;
    nneighbors      = 5;
end

if tiedflag
    ekernels = flipud(dkernels);
end


Xsize       = size(X);
Ksize       = floor(sqrt(size(dkernels,1)));
Zsize       = Xsize(1) - Ksize + 1;
nhidmaps    = length(hbias);
ninputs     = size(X,3);

if gpuDeviceCount
    encoder     = gpuArray.zeros(Zsize, Zsize, nhidmaps);
    decoder     = gpuArray.zeros(Xsize);
    errormap    = gpuArray.zeros(size(encoder));
    gdbh        = gpuArray.zeros(nhidmaps,1);
else
    encoder     = zeros(Zsize, Zsize, nhidmaps);
    decoder     = zeros(Xsize);
    errormap    = zeros(size(encoder));
    gdbh        = zeros(nhidmaps,1);
end

for i = 1:nhidmaps  
    ek{i} = reshape(ekernels(:,(i-1)*ninputs+1:i*ninputs), Ksize, Ksize,ninputs);
    dk{i} = reshape(dkernels(:,(i-1)*ninputs+1:i*ninputs), Ksize, Ksize,ninputs);
end

%%% feed-forward
for i = 1:nhidmaps
    tek = reshape(flipud(ek{i}(:)),size(ek{i}));
    encoder(:,:,i) = convn(X, tek, 'valid') + hbias(i); % this is correlation operation 
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

switch acttype
    case 'max'
        [~, gmap] = feval(@maxAct, map);
    case 'tanh'
        [~, gmap] = feval(@tanhAct, map);
    case 'sigmoid'
        [~, gmap] = feval(@sigmoidAct, map);
    otherwise
        return;
end


%%% decoder
for i = 1:ninputs
    tempout = zeros(Xsize(1),Xsize(2));
    for j = 1:nhidmaps
        tdk = reshape(flipud(dk{j}(:)), size(dk{j}));
        tempout = tempout + convn(map(:,:,j), tdk(:,:,end-i+1), 'full');
    end    
    decoder(:,:,i) = tempout;
end
decoder = decoder + obias;

%% backward
% gradient in the decoder module
gddw = [];
delta = (decoder - X);
gdbo = sum(delta(:));
for j = 1:nhidmaps 
    gddw              = horzcat(gddw,...
                        reshape( rot180(convn(delta, rot90(map(:,:,j),2), 'valid')), Ksize^2,[]));
    errormap(:,:,j)   = convn(flipdim(delta,3), dk{j}, 'valid');    
end
errormap = errormap .* gmap;

%%% gradient in the encoder module
gdew = [];
for j = 1:nhidmaps
    terrormap   = errormap(:,:,j);
    gdew        = horzcat(gdew, reshape(convn(X, rot90(terrormap,2), 'valid'), Ksize^2, []));
    gdbh(j)     = sum(terrormap(:));
end 

%% display results
rerror      =  0.5 * sum((decoder(:) - X(:)).^2);
% cost        = rerror + 1/numData * lambda * sum(sqrt(map(:).^2 + 1e-6)) + alpha * sum(sqrt(ekernels(:).^2 + 1e-6)) + 0.5 * beta * sum(ekernels(:).^2);
% disp(['error ' num2str(rerror) ';  cost' num2str(cost) '; sparsity' num2str(numel(find(abs(map(:))<0.01))/numel(map)) ', ' num2str(numel(find(abs(map(:))==0))/numel(map))]);



%% update weights
epsilonw    = params.epsilonw;
epsilonb    = params.epsilonb;
epsilono    = params.epsilono;
winc        = params.winc;
momentum    = params.momentum;

if tiedflag
    winc = momentum * winc -   (1-momentum) * epsilonw  * (gddw + flipud(gdew));
    dkernels = dkernels + winc;
    params.winc = winc;
end

hbias       = hbias   - epsilonb * gdbh;
obias       = obias   - epsilono * gdbo;
end

function X = rot180(X)
tX = flipdim(X,1);
X  = flipdim(tX,2);
end