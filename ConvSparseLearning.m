noutputmaps             = numfeatures1;
if length(imgsize) > 2
    randConnect         = imgsize(3);
else
    randConnect         = 1;
end

tiedparams.lambda       = 0;    % sparsity regularization
tiedparams.alpha        = 0;     % L1 decay
tiedparams.beta         = 0;      % L2 decay
tiedparams.epsilonw     = 1e-3;  %-4
tiedparams.epsilonb     = 1e-4;  %-5
tiedparams.epsilono     = 1e-4;  %-5
tiedparams.momentum     = 0.5;
tiedparams.momentumf    = 0.99;
params                  = tiedparams;
    

kernels                 = InitialKernels(kernelsize1,noutputmaps,randConnect);
Zsize                   = zeros(imgsize(1)-kernelsize1+1, imgsize(2)-kernelsize1+1);  % feature map size
hbias                   = zeros(numfeatures1,1);
obias                   = 0;%zeros(randConnect,1);
params.winc             = zeros(size(kernels));

tiedflag                = 1;
poolstride              = poolstride1;
vneighbors              = vneighbors1;


% learning params
maxepoch                = 50;
count                   = 0;

% dynamic error
derror                  = [];
dmerror                 = []; % smooth error in previous 'dgap' steps.
dgap                    = 10;


batch_model             = 0;
online_model            = 1;
if batch_model
    minibatch           = 5;
    nloop               = ceil(numTrains/minibatch);
else
    nloop               = numTrains;
end

telapsed_total = []; % elapsed time in each epoch
de_ave = []; % average error in each epoch
dme_ave = []; % mean smooth error in each epoch
de_sum = 0; 
dme_sum = 0;

% learning
for epoch = 1:maxepoch
    telapsed_loop = 0;
    fprintf('eopch: %d/%d\t', epoch, maxepoch);
    if numTrains > 500 & online_model
        index = randperm(numTrains, 500);
        nloop = 500;
    else
        index = randperm(numTrains);
    end
    
    for loop = 1:nloop
        
        count = count + 1;
        
        if batch_model == 1
            startIndex = mod((loop-1) * minibatch, numTrains) + 1;
            endIndex = startIndex + minibatch-1;
            if endIndex > numTrains, endIndex = numTrains; end
            nminidata = endIndex - startIndex + 1;
            
            img = reshape(xtrain(:,index(startIndex:endIndex)),imgsize(1),imgsize(2),imgsize(3),nminidata);
        elseif online_model == 1
%             img = squeeze(xtrain(:,:,:,index(loop)));  
            img = reshape(xtrain(:,index(loop)), imgsize);  
            numSamples = 1;
        end
        
        tstart = tic;
        [kernels, hbias, obias, params, ri, error]  = TiedRecstConvNets(img, acttype, kernels,...
            hbias, obias, params, poolstride, tiedflag, vneighbors);
        telapsed = toc(tstart);
        telapsed_loop = telapsed_loop + telapsed;
        
        if count < 200
            params.momentum = count/200 * params.momentumf + (1 - count/200) * params.momentum;
        else
            params.momentum = params.momentumf;
        end
        
        %% error and display
        derror = [derror, sum(error(:))];
        if count > 10
            dmerror = [dmerror, sum(derror(end-dgap+1:end))/dgap]; % the smooth error in 'dgap' steps
%             if ~mod(count, 100), figure(1); plot(dmerror); end 
        end
       
%         if ~mod(count, 100)
%             figure(2);
%             if strcmpi(datasetName,'CIFAR10')
%                 show_centroids(reshape(kernels, size(kernels,1) * 3, [])' * 20, kernelsize1);
%             else
%                 display_network(kernels);  title 'Kernels'
%             end
%             figure(3); subplot(131), imshow(img,[]);  title 'Original image'%imagesc(img); title 'Original image'
%             subplot(132); imshow(ri, []);  title 'Reconstructed image'%imagesc(img+E); title 'Reconstructed image'
%             subplot(133); imshow(img - ri,[]);   title 'Residue image'%imagesc(-E); title 'Residue image'
%         end
        
    end
    
    de_ave_t = (sum(derror(:)) - de_sum)/nloop; % average error in the current epoch
    de_ave = [de_ave, de_ave_t];
    
    de_sum = sum(derror(:));
    if epoch == 1
        dme_ave_t = (sum(dmerror(:)) - dme_sum)/(nloop - 10);
    else
        dme_ave_t = (sum(dmerror(:)) - dme_sum)/nloop;
    end
    dme_ave = [dme_ave, dme_ave_t]; % mean smooth error in the current epoch
    dme_sum = sum(dmerror(:));
    
    fprintf('elapsed time: %f, number of trains: %d\n', telapsed_loop, nloop);
    telapsed_total = [telapsed_total, telapsed_loop];
    
    % break condition
    if epoch > 1 & ((abs(de_ave(end-1) - de_ave(end)) < 0.01) | (abs(dme_ave(end-1) - dme_ave(end)) < 0.01))
        break;
    end
end


save([dictionaryPath filesep str], 'kernels', 'hbias', 'obias', 'poolstride','vneighbors', 'dmerror', 'tiedparams',...
    'derror', 'telapsed_total', 'de_ave', 'dme_ave');

