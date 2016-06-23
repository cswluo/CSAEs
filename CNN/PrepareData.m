function PrepareData(imdb, getBatch, opts, datasetName)
    train = find(imdb.images.set == 1) ;
    ytrain = [];
    bs = 256 ;

    for t=1:bs:numel(train)
        batch_time = tic ;
        batch = train(t:min(t+bs-1, numel(train))) ;
        fprintf('computing average image: processing batch starting with image %d ...', batch(1)) ;
        [temp, label] = getBatch(imdb, batch) ;
        im{t} = temp;
        ytrain = [ytrain; label'];
        batch_time = toc(batch_time) ;
        fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
    end
    xtrain = cat(4, im{:}) ;
    save([opts.dataDir filesep datasetName '_data.mat'], 'xtrain','ytrain') ;
end