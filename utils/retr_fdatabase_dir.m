function [fdatabase] = retr_fdatabase_dir(rt_data_dir)
%=========================================================================
% inputs
% rt_data_dir   -the rootpath for the database. e.g. '../data/caltech101'
% outputs
% database      -a tructure of the dir
%                   .path   pathes for each image file
%                   .label  label for each image file
% written by Jianchao Yang
% Mar. 2009, IFP, UIUC
%=========================================================================

fprintf('dir the fdatabase...');
subfolders = dir(rt_data_dir);

fdatabase = struct();
fdatabase.path = {};         % path for each image feature
fdatabase.label = [];       % class label for each image feature
count = 0;

for i = 1:length(subfolders)
    
    subname = subfolders(i).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        
        count = count + 1;
        frames = dir(fullfile(rt_data_dir, subname, '*.mat'));
        c_num = length(frames);
        fdatabase.label = [fdatabase.label; ones(c_num, 1)*count];
        
        for j = 1:c_num,
            c_path = fullfile(rt_data_dir, subname, frames(j).name);
            fdatabase.path = [fdatabase.path, c_path];
        end;    
    end;
end;
disp('done!');