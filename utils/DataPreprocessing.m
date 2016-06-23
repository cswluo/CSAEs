function [xtrain, varargout] = DataPreprocessing(X,varargin)

if nargin == 2
    dataDir = varargin{1};
elseif nargin == 3
    dataDir = varargin{1};
    dataName = varargin{2};
elseif nargin == 6
    dataDir = varargin{1};
    dataName = varargin{2};
    use_whitening = varargin{5};
    patchNameSize = varargin{3};
    layerName = varargin{4};
else
    dataDir = './';
    dataName = [];
    use_whitening = 1;
    patchNameSize = [];
end

%% preprocessing data

if max(X(:)) > 200
    X = X./255;
end


%======================================================================
% whitening

ind = randi(size(X,2),100,1);
% figure;display_network(X(:,ind)); title 'before whitening'
figure;show_centroids(X(:,ind)',32); title 'before whitening'

if use_whitening
    
    % local normalization
%     X = bsxfun(@minus, X, mean(X,1));
%     X = bsxfun(@rdivide, X, sqrt(var(X,[],1)+eps));
    
    %%=====================================================================
    %%estimate parameter for smooth filteringb
    
    sigma = X * X' / size(X, 2);
    [U, S, V] = svd(sigma, 0);
    figure;plot(diag(S));
    
    %%=====================================================================
    
    % whitening
    C = cov(X');
    M = mean(X,2);
    [V,D] = eig(C);
    P = V * diag(sqrt(1./(diag(D+0.1)))) * V';
    X = P * bsxfun(@minus, X, M);
%     figure;display_network(X(:,ind)); title 'after whitening'
figure;show_centroids(X(:,ind)',32); title 'after whitening'
else
    %%=================================================================
    %%=Gaussian variable
    
    %         X = bsxfun(@minus, X, mean(X,2));
    %         X = bsxfun(@rdivide, X, std(X,[],2)+eps);
    
    %%=================================================================
    %%= Truncate to +/-3 standard deviations and scale to -1 to 1
    
    %         pstd = 3 * std(X(:));
    %         X = max(min(X, pstd), -pstd) / pstd;
    %         % Rescale from [-1,1] to [0.1,0.9]
    % %         X = (X + 1) * 0.4 + 0.1;
    %         X = (X + 1) * 0.5;
    
    %%=================================================================
    %%= remove DC component
    
    
    %         pstd = 3 * std(X(:));
    %         X = max(min(X, pstd), -pstd) / pstd;
    %         X = (X + 1) * 0.5;      % Rescale from [-1,1] to [0.1,0.9]
    
    
end
%     figure;display_network(X(:,ind)); title 'after whitening'

%%=====================================================================
%%save data

xtrain = X;
clear X
if use_whitening
    save([dataDir filesep dataName '_' patchNameSize '_' layerName '_train'], 'xtrain','M','P');
else
    save([dataDir filesep dataName '_' patchNameSize '_' layerName '_train'], 'xtrain');
end

%%=====================================================================
%%output data

if nargout == 3
    varargout{1} = M;
    varargout{2} = P;
end
end