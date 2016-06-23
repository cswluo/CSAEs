if exist([dictionaryPath filesep str '.mat'], 'file');
    load([dictionaryPath filesep str]);
    return;
end
poolstride              = poolstride2;
vneighbors              = vneighbors2;
maxepoch                = 10;
dgap                    = 10;


if exist('tiedparams', 'var')
    clear tiedparams;
end
tiedparams.lambda       = 0;    % sparsity regularization
tiedparams.alpha        = 0;     % L1 decay
tiedparams.beta         = 0;      % L2 decay
tiedparams.epsilonw     = 1e-3;  % for unnormalized MNIST 1e-3
tiedparams.epsilonb     = 1e-4;  % for unnormalized MNIST 1e-3
tiedparams.epsilono     = 1e-4;  % for unnormalized MNIST 1e-2
tiedparams.momentum     = 0.5;
tiedparams.momentumf    = 0.99;
params                  = tiedparams;

noutputmaps             = numfeatures2;
randConnect             = wiredConnect;
kernels                 = InitialKernels(kernelsize2, noutputmaps, randConnect);
count                   = 0;
tiedflag                = 1;
plf                     = 0;
for pl = 1:length(net.layers)
    if strcmpi(net.layers{pl}.type,'conv')
        plf = plf + 1;
        if plf == 2, break; end
    end
end
padflag                 = net.layers{pl}.pad;


ngroup                  = nummaps/randConnect;
groups                  = noutputmaps/ngroup;   % number of filters in each group
for i = 1:noutputmaps
    ind = ceil(i/groups);
    cnt(:,i) = (ind - 1) * randConnect + 1 : ind * randConnect;
end
params.winc             = zeros(size(kernels,1), randConnect * groups);
params                  = repmat(params,ngroup,1);
hbias                   = zeros(noutputmaps,1);
obias                   = zeros(ngroup,1);
derror                  = cell(ngroup,1);
dmerror                 = cell(ngroup,1);

nloop                   = numTrains;
if nloop > 5000, nloop  = 5000; end;

for epoch = 1:maxepoch
    
    fprintf('epoch: %d/%d\n', epoch, maxepoch);
    index = randperm(numTrains);
    
    
    %%% load data from disk    
    if strcmpi(datasetName, 'Caltech101') | strcmpi(datasetName, 'Caltech256')
        utrain = zeros(mapsize(1)+1,mapsize(2)+1,mapsize(3),nloop);
        for ut = 1:nloop
            load(database.path{index(ut)});
            if any(padflag)
                map = padarray(map,[1,1],'post');
            end
            utrain(:,:,:, ut) = map;
        end
    elseif strcmpi(datasetName,'mnist')
        utrain = map(:,:,:,index);
    end
    
    for gup = 1:ngroup
        gkernels{gup} = kernels(:, (gup - 1) * randConnect * groups + 1 : gup * randConnect * groups);
        ghbias{gup} = hbias((gup - 1) * groups + 1 : gup * groups,1);
    end
    
    for gup = 1:ngroup
        
        valindin = (gup - 1) * randConnect + 1 : gup * randConnect;
        tcount = (epoch - 1) * nloop;
        
        for loop = 1:nloop
            tcount = tcount + 1;
            
            if strcmpi(datasetName, 'CIFAR10') | strcmpi(datasetName, 'MNIST');
                inputdata = squeeze(map(:,:,valindin,index(loop)));
            else
                inputdata = squeeze(utrain(:,:,valindin,loop));
            end
            
            [gkernels{gup}, ghbias{gup}, obias(gup), params(gup), ri, error]  = TiedRecstConvNets(...
                inputdata, acttype, gkernels{gup},...
                ghbias{gup}, obias(gup), params(gup), poolstride, tiedflag, vneighbors);
            
            %         [kernels, hbias, obias, params, ri, error]  = TiedRecstConvNets2(inputdata, acttype, kernels,...
            %             hbias, obias, params, cnt, poolstride, tiedflag, vneighbors);
            
            if tcount < 200
                params(gup).momentum = tcount/200 * params(gup).momentumf + (1 - tcount/200) * params(gup).momentum;
            else
                params(gup).momentum = params(gup).momentumf;
            end
            
            %%%
            derror{gup} = [derror{gup}, error];
            if tcount > 10;
                dmerror{gup} = [dmerror{gup}, sum(derror{gup}(end-dgap+1:end))/dgap];
%                 if ~mod(tcount,100), figure(1); plot(dmerror{gup}); drawnow; end
            end
            
%             if ~mod(tcount, 100)
%                 figure(2);  display_network(gkernels{gup}); %subplot(122), display_network(dkernels);    title 'Kernels'
%                 figure(3);  display_network(reshape(inputdata,size(inputdata,1)^2,[]));  title 'Original image'%imagesc(img); title 'Original image'
%                 figure(4);  display_network(reshape(ri,size(ri,1)^2,[]));   title 'Reconstructed image'%imagesc(img+E); title 'Reconstructed image'
%                 figure(5);  display_network(reshape(inputdata - ri,size(inputdata,1)^2,[])); title 'Residue image'%imagesc(-E); title 'Residue image'
%             end
        end
    end
    kernels = cat(2, gkernels{:});
    hbias   = cat(1, ghbias{:});

    
    %%% update learning params, it seems to update these params in every
    %%% loop works better.
    for gup = 1:ngroup
        params(gup).epsilonw     = 0.95^epoch * params(gup).epsilonw;
        params(gup).epsilonb     = 0.95^epoch * params(gup).epsilonb;
        params(gup).epsilono     = 0.95^epoch * params(gup).epsilono;
    end
end

% save results
save([dictionaryPath filesep str], 'kernels','hbias','obias','poolstride','vneighbors','tiedparams','dmerror', '-v7.3');
