function [a da,varargout] = pshkgAct(x,sh)

[xsize(1),xsize(2),nummaps, numdata] = size(x);
a           = zeros(xsize(1),xsize(2),nummaps, numdata);
da          = zeros(xsize(1),xsize(2),nummaps, numdata);
gradbeta    = zeros(xsize(1),xsize(2),nummaps, numdata);
gradb       = zeros(xsize(1),xsize(2),nummaps, numdata);
ncnt        = length(sh.beta);
lep         = 1e-6;

if nummaps == ncnt
    for i = 1:nummaps
        beta        = sh.beta(i);
        b           = sh.b(i);
        tdata       = squeeze(x(:,:,i,:));
        a(:,:,i,:)  = sign(tdata) .* (1/beta * log(exp(beta*b) + exp(beta*sqrt(tdata.^2 + lep)) - 1) - b);
        
        M           = exp(beta*b) + exp(beta*sqrt(tdata.^2 + lep))-1;
        
        % gradient for data
        da(:,:,i,:) = sign(tdata) .* (tdata./sqrt(tdata.^2+lep) .* exp(beta * sqrt(tdata.^2+lep)))./(M+lep);
        
        % gradient for params
        if nargout == 4
            gradbeta(:,:,i,:)    = sign(tdata) .* ((b*exp(beta*b) + sqrt(tdata.^2+lep).*exp(beta*sqrt(tdata.^2+lep)))./(beta*M+lep) - log(M+lep)./(beta^2+lep));
            gradb(:,:,i,:)       = sign(tdata) .* (exp(beta*b)./(M+lep) - 1);
        end        
    end
else
    beta        = sh.beta;
    b           = sh.b;
    a           = sign(x) .* (1/beta * log(exp(beta*b) + exp(beta*sqrt(x.^2 + lep)) - 1) - b);
    
    M           = exp(beta*b) + exp(beta*sqrt(x.^2 + lep))-1;    
    % gradient for data
    da          = sign(x) .* (x./sqrt(x.^2+lep) .* exp(beta * sqrt(x.^2+lep)))./(M+lep);
    
    % gradient for params
    if nargout == 4
        gradbeta    = sign(x) .* ((b*exp(beta*b) + sqrt(x.^2+lep).*exp(beta*sqrt(x.^2+lep)))./(beta*M+lep) - log(M+lep)./(beta^2+lep));
        gradb       = sign(x) .* (exp(beta*b)./(M+lep) - 1);
    end
    
end

if nargout == 4
    varargout{1}    = gradbeta;
    varargout{2}    = gradb;
end

end