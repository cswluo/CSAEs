function [info] = cnn_test(net, xtest, ytest, varargin)

opts.batchSize = 256 ;
opts.useGpu = false ;
opts.errorType = 'multiclass' ;
opts.expDir = 'data/exp' ;
opts.test = [] ;
opts.imgsize = [];

opts.conserveMemory = true ;
opts.sync = true ;
opts.prefetch = false ;

opts = vl_argparse(opts, varargin) ;

imgsize     = opts.imgsize;

if ~exist(opts.expDir), mkdir(opts.expDir) ; end
if isempty(opts.test), opts.test = 1:length(ytest) ; end
if isnan(opts.test), opts.test = [] ; end

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  for i=1:numel(net.layers)
    if ~strcmp(net.layers{i}.type,'conv'), continue; end
    net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
    net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
  end
end


%% test
info.objective = 0 ;
info.error = 0;
info.topFiveError = 0;

test = opts.test ;

for t=1:opts.batchSize:numel(test)
    
    res = [];
    batch = test(t:min(t+opts.batchSize-1, numel(test))) ;
 
    im = reshape(xtest(:,batch), imgsize(1), imgsize(2), imgsize(3), length(batch));
    labels = ytest(batch);
    
    if opts.useGpu
        im = gpuArray(im) ;
    end
    
    % prediction
    net.layers{end}.class = labels ;
    res = vl_simplenn(net, im, [], res, ...
        'disableDropout', true, ...
        'conserveMemory', opts.conserveMemory, ...
        'sync', opts.sync);
    
    info = updateError(opts, info, net, res) ;
      
 end
   
 info.objective = info.objective / numel(test) ;
 info.error = info.error / numel(test)  ;
 info.topFiveError = info.topFiveError / numel(test) ;
  
end



%%
function info = updateError(opts, info, net, res)

predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

labels = net.layers{end}.class ;
info.objective = info.objective + sum(double(gather(res(end).x))) ;
switch opts.errorType
  case 'multiclass'
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
    info.error = info.error +....
      sum(sum(sum(error(:,:,1,:))))/n ;
    info.topFiveError = info.topFiveError + ...
      sum(sum(sum(min(error(:,:,1:5,:),[],3))))/n ;
  case 'binary'
    error = bsxfun(@times, predictions, labels) < 0 ;
    info.error = info.error + sum(error(:))/n ;
end
end