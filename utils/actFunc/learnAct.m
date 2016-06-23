function [a da, varargout] = learnAct(x,sh)

xsize = size(x);
a = zeros(xsize);

if isstruct(sh)
    beta = sh.beta;
    b = sh.b;
    t = x;    
    a = sign(t) .* (1/beta * log(exp(beta*b) + exp(beta*abs(t)) - 1) - b);
elseif iscell(sh)
    for i = 1:xsize(end); 
        beta = sh{i}.beta;
        b = sh{i}.b;
        t = x(:,:,:,i);
        a(:,:,:,i) = sign(t) .* (1/beta * log(exp(beta*b) + exp(beta*abs(t)) - 1) - b);
    end
end

if nargout > 1
    da = zeros(xsize);    
    lep = 1e-6;
    if isstruct(sh)
        beta = sh.beta;
        b = sh.b;
        t = x;
        M = exp(beta*b) + exp(beta*sqrt(t.^2 + lep))-1;
        % gradient for data
        da = sign(t) .* (t./sqrt(t.^2+lep) .* exp(beta * sqrt(t.^2+lep)))./(M+lep);
        % gradient for params
        if nargout == 4
            gradBeta    = sign(t) .* ((b*exp(beta*b) + sqrt(t.^2+lep).*exp(beta*sqrt(t.^2+lep)))./(beta*M+lep) - log(M+lep)./(beta^2+lep));
            gradb       = sign(t) .* (exp(beta*b)./(M+lep) - 1);
            varargout{1} = gradBeta;
            varargout{2} = gradb;
        end
    elseif iscell(sh)
        beta = sh{i}.beta;
        b = sh{i}.b;
        for i = 1:xsize(end)
            t = x(:,:,:,i);
            M = exp(beta*b) + exp(beta*sqrt(t.^2 + lep))-1;
            % gradient for data
            da(:,:,:,i) = sign(t) .* (t./sqrt(t.^2+lep) .* exp(beta * sqrt(t.^2+lep)))./(M+lep);
        end
    end          
end

end