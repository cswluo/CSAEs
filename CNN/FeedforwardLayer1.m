if ~isdir(fullfile(dataPath, datasetName, currentLayer)) & any(strcmpi(datasetName, {'Caltech101','Caltech256'})) 
    mkdir(fullfile(dataPath, datasetName, currentLayer));
end

if feedforwardFlag
    if any(strcmpi(datasetName, {'MNIST','CIFAR10'}))  
        mapsize         = [12,12];
        nmaps           = numfeatures1;
        map             = zeros(mapsize(1), mapsize(2), nmaps, numTrains);
        minibatch       = 500;
        nloop           = ceil(numTrains/minibatch); 
        for i = 1:nloop
            startIndex = mod((i-1) * minibatch, numTrains) + 1;
            endIndex = startIndex + minibatch-1;
            if endIndex > numTrains, endIndex = numTrains; end          
            nminidata = endIndex - startIndex + 1;
            trainx = reshape(xtrain(:,startIndex:endIndex),imgsize(1), imgsize(2), 1, nminidata);
            mapArray = FirstBlockOutput(net,trainx,'nlayerout',3);
            map(:,:,:,startIndex:endIndex) = mapArray;
        end
        clear mapArray
        if gpuDeviceCount
            map = gather(map);
        end
        filepath = fullfile(dataPath, datasetName);
        if ~isdir(filepath)
            mkdir(filepath);
        end
        dststr = [filepath filesep currentLayer '.mat'];
        savemap(dststr, map);        
    elseif any(strcmpi(datasetName, {'Caltech256','Caltech101'}))
        minibatch       = 100;
        nloop           = ceil(numTrains/minibatch);  
        
        for i = 1:nloop
            startIndex = mod((i-1) * minibatch, numTrains) + 1;
            endIndex = startIndex + minibatch-1;
            if endIndex > numTrains, endIndex = numTrains; end          
            nminidata = endIndex - startIndex + 1;
            trainx = reshape(xtrain(:,startIndex:endIndex),imgsize(1), imgsize(2), 1, nminidata);
            mapArray = FirstBlockOutput(net,trainx,'nlayerout',4); 
           
            
            [folderIndex,~] = find(ytrain(:,startIndex:endIndex));
            folderIndex = unique(folderIndex);
            for j = 1:length(folderIndex)
                if ~isdir(fullfile(dataPath, datasetName, currentLayer, num2str(folderIndex(j)) ));
                    mkdir(fullfile(dataPath, datasetName, currentLayer, num2str(folderIndex(j)) ));
                end
            end
            
            ct = 1;
            for k = startIndex:endIndex
                map         = squeeze(mapArray(:,:,:,ct)); ct = ct + 1;
                [~, name]   = fileparts(imgpath{k});
                filepath    = fullfile(dataPath, datasetName, currentLayer, num2str(find(ytrain(:,k))), name);
                dststr      = [filepath '.mat'];
                savemap(dststr, map);
            end
            if ~mod(i,1000), fprintf('%d/%d\n',minibatch*i,numTrains); end
        end
    end

end
