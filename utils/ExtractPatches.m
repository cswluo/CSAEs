function database = ExtractPatches(dataDir, patchSize, stride, maxImSize, dataset, varargin) 

    if nargin == 6
        imPatch  = varargin{1};
        dataPath = ['./data' filesep dataset filesep imPatch];       % directory for saving SIFT descriptors
        feaPath  = ['./feature' filesep dataset filesep imPatch];    % directory for saving final image features
        dicPath  = ['./dictionary' filesep dataset filesep imPatch];
    else
        dataPath = ['./data' filesep dataset];       % directory for saving SIFT descriptors
        feaPath  = ['./feature' filesep dataset];    % directory for saving final image features
        dicPath  = ['./dictionary' filesep dataset];
    end    

    database = [];
    database.imnum = 0; % total image number of the database
    database.cname = {}; % name of each class
    database.label = []; % label of each class
    database.path = {}; % contain the pathes for each image of each class
    database.nclass = 0;
		
    if strcmp(dataset,'Caltech101')
        %% caltech101 dataset
        disp('Extracting Image Patches...');
        subfolders = dir(dataDir);
        
        for i = 1:length(subfolders),
            subname = subfolders(i).name;
            
            if ~strcmp(subname, '.') & ~strcmp(subname, '..')
                
                database.nclass = database.nclass + 1;
                database.cname{database.nclass} = subname;
                
                % images in each subfolder
                frames = dir(fullfile(dataDir, subname, '*.jpg'));
                
                c_num = length(frames);
                database.imnum = database.imnum + c_num;
                database.label = [database.label; ones(c_num, 1)*database.nclass];
                
                imagePath = fullfile(dataPath, subname);
                if ~isdir(imagePath),
                    mkdir(imagePath);
                end;
                
                for j = 1:c_num,
                    imgpath = fullfile(dataDir, subname, frames(j).name);
                    
                    I = imread(imgpath);
                    if ndims(I) == 3,
                        I = im2double(rgb2gray(I));
                    else
                        I = im2double(I);
                    end;
                    
                    [im_h, im_w] = size(I);
                    
                    if max(im_h, im_w) > maxImSize,
                        I = imresize(I, maxImSize/max(im_h, im_w), 'bicubic');
                        [im_h, im_w] = size(I);
                    end;
                    
                    % make grid sampling SIFT descriptors
                    remX = mod(im_w-patchSize,stride);
                    offsetX = floor(remX/2)+1;
                    remY = mod(im_h-patchSize,stride);
                    offsetY = floor(remY/2)+1;
                    
                    [gridX,gridY] = meshgrid(offsetX:stride:im_w-patchSize+1, offsetY:stride:im_h-patchSize+1);
                    
                    fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches\n', ...
                        frames(j).name, im_w, im_h, size(gridX, 2), size(gridX, 1), numel(gridX));
                    
                    % extract patches
                    num_patches = numel(gridX);
                    imagePatches = zeros(patchSize^2, num_patches);
                    for k = 1:num_patches
                        % find window of pixels that contributes to this descriptor
                        x_lo = gridX(k);
                        x_hi = gridX(k) + patchSize - 1;
                        y_lo = gridY(k);
                        y_hi = gridY(k) + patchSize - 1;
                        imagePatches(:,k) = reshape(I(y_lo:y_hi,x_lo:x_hi),patchSize^2,1);
                    end
                    
                    patchSet.feaArr = imagePatches;
                    patchSet.x = gridX(:);
                    patchSet.y = gridY(:);
                    patchSet.width = im_w;
                    patchSet.height = im_h;
                    
                    [pdir, fname] = fileparts(frames(j).name);
                    fpath = fullfile(dataPath, subname, [fname, '.mat']);
                    
                    save(fpath, 'patchSet');
                    database.path = [database.path, fpath];
                end;
            end;
        end
    else
    end
    
end

