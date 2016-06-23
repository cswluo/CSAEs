function [a, comp, da] = compMaxAct(x)

xsize   = size(x);
index   = zeros(xsize);
comp    = zeros(xsize);
a       = zeros(xsize(1),xsize(2),xsize(end));

for i = 1:xsize(end)
    [tmap, tind] = max(x(:,:,:,i),[],3);
    tind(tmap<=0) = 0;
    a(:,:,i) = max(tmap,0);
    for j = 1:xsize(3)
        index(:,:,j,i) = tind == j;
    end   
end
comp(logical(index)) = x(logical(index));
% comp = max(0,x);
da = index;

end