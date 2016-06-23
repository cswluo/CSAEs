function [patches] = RandomLayerSamples(database, patch_size, num_patches) 

num_data = database.imnum;
load(database.path{1})
mapsize = size(map);
clear map
patches = zeros(patch_size, patch_size, mapsize(3), num_patches);

parfor i = 1:num_patches
    
    if (mod(i,100) == 0),  fprintf('.'); end
    if (mod(i,10000) == 0), fprintf('Extracting patch: %d / %d\n', i, num_patches); end
    
    ind = mod(i-1,num_data)+1;
    map = load(database.path{ind});
    
    r = random('unid', mapsize(1) - patch_size + 1);
    c = random('unid', mapsize(2) - patch_size + 1);
    patch = map.map(r:r+patch_size-1,c:c+patch_size-1,:);
    patches(:, :, :, i) = patch;
    
end
end