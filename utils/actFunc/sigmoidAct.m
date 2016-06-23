function [a da] = sigmoidAct(x)

a = 1 ./ (1 + exp(-x));
if nargout > 1
    da = a .* (1-a);
end
end