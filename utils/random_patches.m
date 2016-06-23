% random sample patches from Berkley training set
function [varargout] = random_patches(varargin)

    if nargin < 4
        img_dir = 'D:\BaiduDrive\My Database\Images\Berkerly\BSR\BSDS500\data\images\train'; % Berkley training set
        num_patches = 100000;
        patch_size = 16;
        datasetName = [];
    elseif nargin == 4
        img_dir = varargin{1};
        num_patches = varargin{2};
        patch_size = varargin{3}; 
        datasetName = varargin{4};
    elseif nargin == 6
        img_dir = varargin{1};
        num_patches = varargin{2};
        patch_size = varargin{3};
        datasetName = varargin{4};
        dataPath = varargin{5};
        patchNameSize = varargin{6};
    else
        datasetName = [];
    end
    
    patches = zeros(patch_size^2,num_patches);
    total_patches = 0; ct = 1;
    
    
    if strcmp(datasetName, 'BSD500')  %n_images_in_current_folder > 0
        %% images in current folder
        
        n_images_in_current_folder = length(dir(fullfile(img_dir, '*.mat')));
        
        nimg = n_images_in_current_folder;
        images = dir(fullfile(img_dir, '*.mat'));        
        
        for i = 1:num_patches
            idata = mod((i-1), nimg) + 1;
            load(fullfile(img_dir,images(idata).name));
%             img = imread(fullfile(img_dir,images(idata).name));
%             if size(img,3) ~= 1
%                 img = im2double(rgb2gray(img));
%             else
%                 img = im2double(img);
%             end            
%             % image normlization
%             img = img - mean(img(:));
%             img = img./std(img(:));
            
            [h, w] = size(img);
            
            x = random('unid', h - patch_size + 1);
            y = random('unid', w - patch_size + 1);
            patches(:,ct) = reshape(img(x:x+patch_size-1,y:y+patch_size-1),[1,patch_size^2]);
            ct = ct + 1;           
            
        end
        
        assert(ct-1 == num_patches);
        
    elseif strcmp(datasetName, 'Caltech101')      
    %% current folder is parent folder, e.g. Caltech101
    
        subfolders = dir(img_dir);
        totalImages = 0;
        
        temp = 'D:\BaiduDrive\My Database\Classification\CalTech\101_ObjectCategories';
%         temp = 'D:\BaiduDrive\My Database\Classification\CalTech\101_ObjectCategories';
        subfolders_ = dir(temp);
        totalImages_ = 0;
        for i = 3:length(subfolders)
            if length(dir(fullfile(img_dir, subfolders(i).name, '*.mat'))) ~= length(dir(fullfile(temp, subfolders_(i).name, '*.jpg')))
                fprintf('%s\n',subfolders(i).name);
            end
            totalImages = totalImages + length(dir(fullfile(img_dir, subfolders(i).name, '*.mat')));
            totalImages_ = totalImages_ + length(dir(fullfile(temp, subfolders_(i).name, '*.jpg')));
        end
        patch_per_image = floor(num_patches/totalImages);
      
        for i = 1:length(subfolders)

            if ~strcmp(subfolders(i).name, '.') & ~strcmp(subfolders(i).name, '..')
                
                images = dir(fullfile(img_dir, subfolders(i).name, '*.mat'));
                fprintf('\ncurrent processing: %d/%d, num_images = %d',i-2,length(subfolders)-2,length(images));
                
                if i == length(subfolders)
                    restPatches = num_patches - total_patches;
                    patch_per_image = round(restPatches/length(images));
                end
                
                for j = 1:length(images)
                    
                    if j == length(images) & i == length(subfolders)
                        patch_per_image = num_patches - total_patches;
                    end
                    
                    if ~mod(j,10), fprintf('.'); end
                    
                    load(fullfile(img_dir, subfolders(i).name, images(j).name));
                    
                    if size(img,3) ~= 1
                        img = double(rgb2gray(img));
                    else
                        img = double(img);
                    end
                    
                    [h, w] = size(img);
                 
                    k = 1;
                    while k  <= patch_per_image
                        x = random('unid', h - patch_size + 1);
                        y = random('unid', w - patch_size + 1);
                        temp = img(x:x+patch_size-1,y:y+patch_size-1);
                        if std(temp(:)) > 1e-3, 
                            patches(:,ct) = reshape(temp,[patch_size^2, 1]);
                            ct = ct + 1; k = k + 1;
                        end;                        
                    end
                    total_patches = total_patches + patch_per_image;
                end
                
            end
        end
        
        assert(total_patches == num_patches && ct-1 == num_patches);
    elseif strcmp(datasetName, 'Scene15')      
    %% current folder is parent folder, e.g. Caltech101
        [database] = retr_database_dir(img_dir);
        ndata = database.imnum;    
          
        for i = 1:num_patches
            idata = mod((i-1), ndata) + 1;
            load(database.path{idata}); 
            if size(img,3) ~= 1, img = double(rgb2gray(img));else img = double(img);end            
            [h, w] = size(img);
            dF = 1;
            while dF
                x = random('unid', h - patch_size + 1);
                y = random('unid', w - patch_size + 1);
                temp = img(x:x+patch_size-1,y:y+patch_size-1);
                if std(temp(:)) > 1e-3,
                    patches(:,ct) = temp(:);
                    ct = ct + 1; dF = 0;
                end;
            end
        end

        assert(ct-1 == num_patches);
    elseif strcmp(datasetName, 'MNIST')
        %% load MNIST dataset
        
        converter;
        makebatches;     
        
        if isempty(num_patches)            
            rd = randperm(size(train_x,2));
            xtrain = train_x(:,rd(1:5/6 * length(rd)));
            ytrain = train_y(:,rd(1:5/6 * length(rd)));
            xval = train_x(:,rd(5/6 * length(rd) + 1:end));
            yval = train_y(:,rd(5/6 * length(rd) + 1:end));
            save([dataPath filesep datasetName '_' patchNameSize '_train'], 'xtrain','ytrain');
            save([dataPath filesep datasetName '_' patchNameSize '_valid'], 'xval','yval');
            save([dataPath filesep datasetName '_' patchNameSize '_test'], 'xtest','ytest');
            patches = xtrain;
        else
            ndata = size(train_x,2);
            h = sqrt(size(train_x,1));
            w = h;
            for i = 1:num_patches
                idata = mod((i-1), ndata) + 1;
                img = reshape(train_x(:,idata),h,w);

                x = random('unid', h - patch_size + 1);
                y = random('unid', w - patch_size + 1);
                patches(:,ct) = reshape(img(x:x+patch_size-1,y:y+patch_size-1),[patch_size^2, 1]);
                ct = ct + 1;

                if ~mod(i,1000), fprintf('.');end;
                if ~mod(i,6000), fprintf('\n'); end;
            end
            
            assert(ct-1 == num_patches);
        end
   
    elseif strcmp(datasetName,'CIFAR10')
        %% Load CIFAR training data
        fprintf('Loading training data...\n');
        f1=load([img_dir '/data_batch_1.mat']);
        f2=load([img_dir '/data_batch_2.mat']);
        f3=load([img_dir '/data_batch_3.mat']);
        f4=load([img_dir '/data_batch_4.mat']);
        f5=load([img_dir '/data_batch_5.mat']);
        
        trainX = double([f1.data; f2.data; f3.data; f4.data; f5.data]);
        trainY = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!
        clear f1 f2 f3 f4 f5;
        CIFAR_DIM = [32 32 3];
        
        for i=1:num_patches
            if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, num_patches); end
            
            r = random('unid', CIFAR_DIM(1) - patch_size + 1);
            c = random('unid', CIFAR_DIM(2) - patch_size + 1);
            patch = reshape( trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM );
            patch = patch(r:r+patch_size-1,c:c+patch_size-1,:);
            patch = rgb2gray(patch./255);
            patches(:,i) = patch(:);
        end
        
    end
    
 
    if nargout >= 1
        varargout{1} = patches;
    end
end