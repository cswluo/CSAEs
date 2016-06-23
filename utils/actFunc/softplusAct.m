function [a da] = softplusAct(x)
a = log(1 + exp(x));
if nargout > 1
    da = exp(x)./(1 + exp(x));
end
end