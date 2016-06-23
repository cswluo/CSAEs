function [net,info] = cnn_train_mnist(net, xtrain, ytrain, varargin)

opts.datainfo.datasetName = [];
opts.datainfo.numClass = 10;
opts.datainfo.imgsize = [];
opts.datainfo.trNum = 30;
opts.datainfo.train = [] ;

opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorType = 'multiclass' ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = 'data/exp' ;
opts.numEpochsPlus = 5;

opts.conserveMemory = false ;
opts.sync = true ;
opts.prefetch = false ;
opts.plotDiagnostics = false ;

opts.times = [];
opts = vl_argparse(opts, varargin) ;

imgsize     = opts.datainfo.imgsize;

if ~exist(opts.expDir), mkdir(opts.expDir) ; end
if isempty(opts.datainfo.train), opts.datainfo.train = 1:length(ytrain) ; end
if isnan(opts.datainfo.train), opts.datainfo.train = [] ; end


%% network initialization
for i=1:numel(net.layers)
  if ~strcmp(net.layers{i}.type,'conv'), continue; end
    net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters)) ;
    net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases)) ;
  if ~isfield(net.layers{i}, 'filtersLearningRate')
    net.layers{i}.filtersLearningRate = 1 ;
  end
  if ~isfield(net.layers{i}, 'biasesLearningRate')
    net.layers{i}.biasesLearningRate = 1 ;
  end
  if ~isfield(net.layers{i}, 'filtersWeightDecay')
    net.layers{i}.filtersWeightDecay = 1 ;
  end
  if ~isfield(net.layers{i}, 'biasesWeightDecay')
    net.layers{i}.biasesWeightDecay = 1 ;
  end
end

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  for i=1:numel(net.layers)
    if ~strcmp(net.layers{i}.type,'conv'), continue; end
    net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
    net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
  end
end


%% train and validate

rng(0) ;

if opts.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.train.speed = [] ;



lr = 0 ;
res = [] ;
modelPath = fullfile(opts.expDir, 'net-pre-train-epoch-%d.mat') ;
load(sprintf(modelPath, opts.numEpochs), 'net', 'info')
for epoch=1:opts.numEpochsPlus
  prevLr = lr ;
  lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % fast-forward to where we stopped
  modelPath = fullfile(opts.expDir, 'net-pre-train-base%d-epoch-%d.mat') ;
  modelFigPath = fullfile(opts.expDir, 'net-pre-train-all.pdf') ;
  if opts.continue 
    if exist(sprintf(modelPath, epoch),'file'), continue ; end
  	if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(sprintf(modelPath, opts.numEpochs, epoch-1), 'net', 'info') ;
    end
  end


  train = opts.datainfo.train(randperm(numel(opts.datainfo.train))) ;
  
  
  info.train.objective(end+1) = 0 ;
  info.train.error(end+1) = 0 ;
  info.train.topFiveError(end+1) = 0 ;
  info.train.speed(end+1) = 0 ;


  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
    for l=1:numel(net.layers)
      if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
      net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum ;
      net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum ;
    end
  end
  
  
  %% train
  for t=1:opts.batchSize:numel(train)
    % get next image batch and labels
    batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
    batch_time = tic ;

    im = reshape(xtrain(:,batch), imgsize(1), imgsize(2), imgsize(3), length(batch));
    labels = ytrain(batch);

    if opts.useGpu
      im = gpuArray(im) ;
    end

    % backprop
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, one, res, ...
      'conserveMemory', opts.conserveMemory, ...
      'sync', opts.sync);

    % gradient step
    for l=1:numel(net.layers)
      if ~strcmp(net.layers{l}.type, 'conv'), continue ; end

      net.layers{l}.filtersMomentum = ...
        opts.momentum * net.layers{l}.filtersMomentum ...
          - (lr * net.layers{l}.filtersLearningRate) * ...
          (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
          - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l).dzdw{1} ;

      net.layers{l}.biasesMomentum = ...
        opts.momentum * net.layers{l}.biasesMomentum ...
          - (lr * net.layers{l}.biasesLearningRate) * ....
          (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
          - (lr * net.layers{l}.biasesLearningRate) / numel(batch) * res(l).dzdw{2} ;

      net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
      net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
    end
  

    % print information
%     fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
%         fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;

    batch_time = toc(batch_time) ;
    speed = numel(batch)/batch_time ;
    info.train = updateError(opts, info.train, net, res, batch_time) ;
% 
%     fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
%     n = t + numel(batch) - 1 ;
%     fprintf(' err %.1f err5 %.1f', ...
%       info.train.error(end)/n*100, info.train.topFiveError(end)/n*100) ;
%     fprintf('\n') ;

    % debug info
    if opts.plotDiagnostics
      figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
    end
    
  end
   
  
  %% save
  info.train.objective(end) = info.train.objective(end) / numel(train) ;
  info.train.error(end) = info.train.error(end) / numel(train)  ;
  info.train.topFiveError(end) = info.train.topFiveError(end) / numel(train) ;
  info.train.speed(end) = numel(train) / info.train.speed(end) ; 

  save(sprintf(modelPath,opts.numEpochs, epoch), 'net', 'info') ;
  

  figure(1) ; clf ;
  subplot(1,2,1) ;
  axiepoch = opts.numEpochs + epoch;
  semilogy(1:axiepoch, info.train.objective, 'k') ; hold on ;
  xlabel('training epoch') ; ylabel('energy') ;
  grid on ;
  h=legend('train') ;
  set(h,'color','none');
  title('objective') ;
  subplot(1,2,2) ;
  switch opts.errorType
    case 'multiclass'
      plot(1:axiepoch, info.train.error, 'k') ; hold on ;
      plot(1:axiepoch, info.train.topFiveError, 'k--') ;
      h=legend('train','train-5') ;
    case 'binary'
      plot(1:epoch, info.train.error, 'k') ; hold on ;
      h=legend('train') ;
  end
  grid on ;
  xlabel('training epoch') ; ylabel('error') ;
  set(h,'color','none') ;
  title('error') ;
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
  
end
end


%%
function info = updateError(opts, info, net, res, speed)

predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

labels = net.layers{end}.class ;
info.objective(end) = info.objective(end) + sum(double(gather(res(end).x))) ;
info.speed(end) = info.speed(end) + speed ;
switch opts.errorType
  case 'multiclass'
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
    info.error(end) = info.error(end) +....
      sum(sum(sum(error(:,:,1,:))))/n ;
    info.topFiveError(end) = info.topFiveError(end) + ...
      sum(sum(sum(min(error(:,:,1:5,:),[],3))))/n ;
  case 'binary'
    error = bsxfun(@times, predictions, labels) < 0 ;
    info.error(end) = info.error(end) + sum(error(:))/n ;
end
end
