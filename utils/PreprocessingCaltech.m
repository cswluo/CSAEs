function PreprocessingCaltech(src,dst)

fSize = 7;
kernel = make_kernel(fSize);
maxImSize = 102;

subfolders = dir(src);

for i = 1:length(subfolders)
    
    if ~strcmp(subfolders(i).name, '.') & ~strcmp(subfolders(i).name, '..')
        
        folder_flag = subfolders(i).isdir;
        
        if folder_flag
            mkdir([dst filesep subfolders(i).name]);
            dstPath = [dst filesep subfolders(i).name];
            
            images = dir(fullfile(src, subfolders(i).name, '*.jpg'));
            fprintf('\ncurrent processing: %d/%d, num_images = %d',i-2,length(subfolders)-2,length(images));
        
            % for each subfolder
            for j = 1:length(images)
                
                if ~mod(j,10), fprintf('.'); end
                
                img = imread(fullfile(src, subfolders(i).name, images(j).name));
                if size(img,3) ~= 1
                    img = double(rgb2gray(img));
                else
                    img = double(img);
                end
                
                %             figure(1); imshow(img,[]);
                % resize image
                img = resize_im(img,maxImSize);
                
                img = imPreProcess(img,kernel);
                %             figure(2); imshow(img,[]);
                [~,name] = fileparts(images(j).name);
                save([dstPath filesep name], 'img');
                
            end
        else
            dstPath = dst;
            if ~mod(i,10), fprintf('.'); end
           
            img = imread(fullfile(src, subfolders(i).name));
            if size(img,3) ~= 1
                img = double(rgb2gray(img));
            else
                img = double(img);
            end
            
            %             figure(1); imshow(img,[]);
            % resize image
            img = resize_im(img,maxImSize);
            
            img = imPreProcess(img,kernel);
            %             figure(2); imshow(img,[]);
            [~,name] = fileparts(subfolders(i).name);
            save([dstPath filesep name], 'img'); 
           
        end      
    end
end

end


function pim = imPreProcess(img,k)

% Processes a given image (img is supposed to be a grayscale image)
% k is the weighting kernel that will be used in local neighborhoods

dim = double(img);
%==========================================================================
% 1. subtract the mean and divide by the standard deviation
mn = mean2(dim);
sd = std2(dim);

dim = dim - mn;
dim = dim / sd;

%==========================================================================
% 2. calculate local mean and std divide each pixel by local std if std>1

lmn     = conv2(dim,k,'valid');
lmnsq   = conv2(dim.^2,k,'valid');
lvar    = lmnsq - lmn.^2;
lvar(lvar<0) = 0; % avoid numerical problems
lstd    = sqrt(lvar);
lstd(lstd<1) = 1;

shifti = floor(size(k,1)/2)+1;
shiftj = floor(size(k,2)/2)+1;

% since we do valid convolutions
dim = dim(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1);
dim = dim - lmn;
dim = dim ./ lstd;

%==========================================================================
% 3. pad with zeros

sz = size(dim);
shift = floor(( max(sz) - sz) / 2);
pim = zeros(max(sz));
pim(1+shift(1):shift(1)+sz(1),1+shift(2):shift(2)+sz(2)) = dim;
end


function imres = resize_im(img,sz)

% input has to be grayscale image
% resize the longer side of input image to sz

szim = size(img);
[maxs,maxi] = max(szim);
szn = [NaN NaN];

szn(maxi) = sz;
imres = imresize(img,szn,'bicubic');
end
