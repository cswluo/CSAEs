function [a da] = linearAct(x)
a = x;
if nargout > 1
    da = ones(size(a));
end
end