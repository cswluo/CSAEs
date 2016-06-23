function [a da] = tanhAct(x, varargin)
if nargin == 1
    a = tanh(x);
    if nargout > 1
        da = (1-a) .* (1+a);
    end
else
    % recommend by Y. LeCun
    a = 1.7159 * tanh(2/3 * x);
    if nargout > 1
        da = 2/3 .* (1 - a) .* (1 + a);
    end
end