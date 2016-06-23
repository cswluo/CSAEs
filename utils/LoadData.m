function [xtrain, varargout] = LoadData(varargin)

    if nargin == 1
        dataset_name = varargin{1};
    elseif nargin == 6
        dataset_name = varargin{1};
        data_dir = varargin{2};
        patch_size = varargin{3};
        num_patches = varargin{4};        
        dataPath = varargin{5};
        patchNameSize = varargin{6};
    else
        data_dir = pwd;
        patch_size = 16;
        num_patches = 100000;
        dataset_name = 'default';
        dataPath = [];
        patchNameSize = 'default';
        
    end
    
    if nargin == 1
        load(dataset_name);
    end
    
    %% generate or load data
    %%===================================================================== load caltech dataset
    if nargin >= 5
        if strcmpi(dataset_name,'MNIST')
            xtrain = random_patches(data_dir, num_patches, patch_size, dataset_name, dataPath, patchNameSize);
        else
            xtrain = random_patches(data_dir, num_patches, patch_size, dataset_name);            
        end
        save([dataPath filesep dataset_name '_' patchNameSize '_org'], 'xtrain');
    end   

    %%===================================================================== Load CIFAR training data
    %     patches = cifar_data(CIFAR_DIR, patch_size, numPatches);
   
    if nargout > 1
        %%================================================================= estimate patch's intrinsic dimention
        varargout{1} = intrinsic_dim(xtrain','EigValue');
    end
    
    
 end