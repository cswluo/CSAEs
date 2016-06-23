function DisplayWeights(ae, varargin)
    
    if nargin == 3
        hierFlag = varargin{1};  
        net_type = varargin{2};
    else
        hierFlag = 0;
        net_type = 'ae';
    end
    
    t = 10;
    
    if hierFlag    % unsuprevise for deep layer        
        W = {};
        for i = 1:length(ae)
            if strcmpi(net_type, 'ae')
                net_size = ae(i).net_size;
                if net_size(2) > 256
                    tolnum = randperm(net_size(2));
                    ind = tolnum(1:256);
                else
                    ind = 1:net_size(2);
                end
                
                if i == 1
                    W{i} = ae(i).W{1};
                    figure; display_network(W{i}(ind,:)'); %title 'ae 1 layer features';
                else
                    WW = WeightsLayer(ae,i,t,'ae');
                    W{i} = WW * W{i-1};
                    figure; display_network(W{i}(ind,:)'); str = sprintf('ae %d layer features.',i);  title(str);
                end
            elseif strcmpi(net_type, 'nn')
                net_size = ae.net_size;
                if exist('ae.layer','var')
                    l = ae.layer;
                else
                    l = length(ae.net_size) + 1;
                end
                for j = 1:l-2
                    
                    if net_size(j+1) > 256
                        tolnum = randperm(net_size(j+1));
                        ind = tolnum(1:256);
                    else
                        ind = 1:net_size(j+1);
                    end
                    
                    if j == 1
                        W{j} = ae.W{j};
                        figure; display_network(W{j}(ind,:)'); title 'nn 1 layer features';
                    else
                        WW = WeightsLayer(ae,j,t,'nn');
                        W{j} = WW * W{j-1};
                        figure; display_network(W{j}(ind,:)'); str = sprintf('nn %d layer features.',j);  title(str);
                    end
                end
            end
        end
    else
        for i = 1:length(ae)            
            net_size = ae(i).net_size;            
            W = ae(i).W{1};
            if net_size(2) >= 256 
                ind = randperm(size(W,1),256);
                W = W(ind,:);
            elseif net_size(2) >= 64 & net_size(2) < 256
                ind = randperm(size(W,1),64);
                W = W(ind,:);
            elseif net_size(2) >= 25 & net_size(2) < 64
                ind = randperm(size(W,1),25);
                W = W(ind,:);
            elseif net_size(2) < 25
                return
            end
            str = sprintf('the %dth AE', i);
            figure; display_network(W'); title(str);
        end
    end
    
end

function WW = WeightsLayer(ae,layer,t,net_type)
    i = layer;
    if strcmpi(net_type,'ae')
        W{i} = ae(i).W{1};
    else strcmpi(net_type,'nn')
        W{i} = ae.W{i};
    end
    [~, ind] = max(abs(W{i}),[],2);
    for j = 1:length(ind)
        sgn(j) = sign(W{i}(j,ind(j)));
    end
    [a, b] = sort(bsxfun(@times, W{i}, sgn'), 2, 'descend');
    weights = bsxfun(@rdivide, exp(a(:,1:t)), sum(exp(a(:,1:t)),2));
    WW = zeros(size(W{i}));
    for j = 1:length(ind)
        WW(j,b(j,1:t)) = weights(j,:);
    end
end