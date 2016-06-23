function ConvSparse_vgg_Caltech101_simple(varargin)


close all
clc;


%%  SET DIRECTORY
addpath(genpath('./utils'));
addpath('./dictionary');
addpath(genpath('./data'));
addpath(genpath('./Caltech'));
addpath('./CNN');
run(fullfile(fileparts(mfilename('fullpath')), './MatConvNet/vl_setupnn.m')) ;


%%% original data path
Caltech101Dir       = '/home/luowei/Datasets/Caltech101';


%%% sava data path
modelPath           = './model';
dataPath            = './data';
dictionaryPath      = './dictionary';
resultPath          = './result';
featurePath         = './feature';

%%% processing dataset
dataDir             = Caltech101Dir;
datasetName         = 'Caltech101';
patchNameSize       = '96';

%%% number of classes
numClass            = 102;     


%%% 
if gpuDeviceCount
    gpud = gpuDevice(2);
    fprintf('success loading GPU.\n');
end

%%%
InitialCNN                 = 1;        % 1, initialize new CNN, need regenerate first layer features, very time comsuming!
convsparsetrain            = 0;        % 1, learn 1st layer decoders
InitialFirLayerWeights     = 1;        % 1, initialize 1st layer weight with unsupervised learned features

InitCALTECHparams


%% PRE-PROCESSING DATA
load Caltech101_whitened
xtrain = single(xtrain);
ytrain = single(ytrain);
numTrains = size(xtrain,2);
if gpuDeviceCount
    xtrain = gpuArray(xtrain);
    ytrain = gpuArray(ytrain);
end
acttype    = 'max'; 
imgsize    = [96 96 1];



%% convolutional sparse learning first layer feature
str = sprintf('covkernel_%s_%s_imgsz%s_mdsz%s-%s_ksz%s-%s_pool%s-%s_vneigb%s', acttype, datasetName, patchNameSize,...
        num2str(numfeatures1), num2str(numfeatures2), num2str(kernelsize1), num2str(kernelsize2),...
        num2str(poolsize1), num2str(poolsize2), num2str(vneighbors1));
     
if convsparsetrain
    ConvSparseLearning;
else
    load([dictionaryPath filesep str]);
end


%% Network initialization
str = sprintf('convnets1_%s_imgsz%s_mdsz%s-%s_ksz%s-%s_pool%s-%s_vneigb%d', datasetName, patchNameSize,...
        num2str(numfeatures1),  num2str(numfeatures2), num2str(kernelsize1), num2str(kernelsize2),...
        num2str(poolsize1),   num2str(poolsize2), ...
        vneighbors1);
if InitialCNN
    net = initializeNetwork_simple;
    save([modelPath filesep str '.mat'], 'net')
else
    load([modelPath filesep str '.mat']);
end



%% Reinitialize the weights of the first layer    
if InitialFirLayerWeights  
    if gpuDeviceCount
        net.layers{1}.filters   = gpuArray(single(reshape(flipud(kernels), size(net.layers{1}.filters))));
        net.layers{1}.biases    = gpuArray(single(hbias'));        
    else
        net.layers{1}.filters   = single(reshape(flipud(kernels), size(net.layers{1}.filters)));
        net.layers{1}.biases    = single(hbias');
    end
    %%% save model
    save([modelPath filesep str], 'net');
    fprintf('Initialize the 1st layer weighs of CNN complete!\n');   
end
clear kernels


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CLASSIFICATION
trNum = 30;
ntimes = 5;
error = zeros(ntimes,1);
top5error = zeros(ntimes,1);
objective = zeros(ntimes,1);
% error(1) = 0.433055091819700;
% top5error(1) = 0.239732888146912;
% objective(1) = 1.901318014205398;

opts.dataDir = 'data/Caltech101' ;
opts.expBaseDir = 'data/Caltech101-baseline' ;
opts.expDir = 'data/Caltech101-baseline/pre1' ;
opts.imdbPath = fullfile(opts.expBaseDir, 'imdb.mat');
opts.lite = false ;
opts.numFetchThreads = 0 ;

opts.train.batchSize = 50 ;
opts.train.numEpochs = 50 ;
opts.train.weightDecay = 0.0005 ;
opts.train.momentum = 0.9 ;
opts.train.continue = false ;
opts.train.useGpu = false ;
opts.train.prefetch = false ;
opts.train.learningRate = [0.01*ones(1, 30) 0.001*ones(1, 15) 0.0001*ones(1,5)] ;
opts.train.expDir = opts.expDir ;
opts.train.errorType = 'multiclass' ;

opts.datainfo.datasetName   = datasetName;
opts.datainfo.numClass      = numClass;
opts.datainfo.imgsize       = imgsize;
opts.datainfo.trNum         = trNum;

[opts, varargin] = vl_argparse(opts, varargin) ;


[~, ytrain] = max(ytrain, [], 1);
backnet = net;
for ltimes = 1:ntimes;
 
    %%% prepare data for each trial   
    indstr = ['data_index_' num2str(ltimes) '.mat'];
    if exist([opts.expDir filesep indstr])
        load([opts.expDir filesep indstr])
        trdata      = xtrain(:,trindex);
        vldata      = xtrain(:,vlindex);
        tsdata      = xtrain(:,tsindex);
        trlabel     = ytrain(trindex); 
        vllabel     = ytrain(vlindex);      
        tslabel     = ytrain(tsindex);
    else
        PrepareTrainTestData
        save([opts.expDir filesep 'data_index_' num2str(ltimes) '.mat'], 'trindex', 'vlindex', 'tsindex');
    end
    
    %%% training and validataion
    if ltimes == 1
    [net,info] = cnn_trainval(net, trdata, trlabel, vldata, vllabel,...
        opts.train, 'conserveMemory', true, 'datainfo', opts.datainfo);
    end
    
    
    %%% train on all training data  
    net = backnet;
    opts.train.numEpochs = 50 ;
    opts.train.batchSize = 50 ;
    opts.train.continue = false ;
    opts.train.learningRate = [0.01*ones(1, 30) 0.001*ones(1, 15) 0.0001*ones(1,5)] ;
    [net,info] = cnn_train(net, [trdata vldata], [trlabel vllabel],...
        opts.train, 'conserveMemory', true, 'datainfo', opts.datainfo, 'times', ltimes);
    save([opts.expDir filesep 'net-1layer-1-final-' num2str(ltimes) '.mat'],'net', 'info');
    
    %%% testing
%     load([opts.expDir filesep 'net-1layer-1-final-' num2str(ltimes) '.mat'], 'net');
    opts.train.batchSize = 100 ;
    info = cnn_test(net, tsdata, tslabel, opts.train, 'imgsize', [96 96 1]);
    save([opts.expDir filesep 'Result-1layer-1-' num2str(ltimes) '.mat'], 'info');
    
    error(ltimes) = info.error;
    top5error(ltimes) = info.topFiveError;
    objective(ltimes) = info.objective;
    
    fprintf('Classification accuracy: %f \n', 1-error(ltimes));
end


Ravg = mean(1-error);
Rstd = std(1-error);

top5Ravg = mean(1-top5error);
top5Rstd = std(1-top5error);
save([opts.expDir filesep 'Performance-1layer-1.mat'], 'Ravg', 'Rstd', 'top5Ravg', 'top5Rstd');
fprintf('===============================================\n');
fprintf('Average classification accuracy: %f\n', Ravg);
fprintf('Standard deviation: %f\n', Rstd);
fprintf('\n');
fprintf('Average top 5 classification accuracy: %f\n', top5Ravg);
fprintf('Top 5 standard deviation: %f\n', top5Rstd);
fprintf('===============================================');

