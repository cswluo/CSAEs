function [a da] = maxAct(x)
a = max(x,0);
if nargout > 1
    da = sign(a);
end
end