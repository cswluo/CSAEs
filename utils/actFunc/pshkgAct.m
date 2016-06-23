function [a da,dbeta,db] = pmaxAct(x,psh)
[xsize(1),xsize(2),nummaps, numdata] = size(x);
beta        = psh.beta;
b           = psh.b;
a           = zeros(xsize(1),xsize(2),nummaps,numdata);
da          = zeros(xsize(1),xsize(2),nummaps,numdata);
dbeta       = zeros(xsize(1),xsize(2),nummaps,numdata);
db          = zeros(xsize(1),xsize(2),nummaps,numdata);

if length(beta) == 1
    a       = 1/beta * log(exp(beta*b) + exp(beta*x)) - b;
    
    if nargout > 1
        M       = exp(beta * b) + exp(beta * x);
        da      = 1./M .* exp(beta * x);
        dbeta   = -1/(beta^2) .* log(M) + 1./(beta*M) .* (b*exp(beta*b) + x.*exp(beta*x));
        db      = 1./M .* exp(beta*b) - 1;
    end
else  
    for i = 1:nummaps
        a(:,:,i,:) = 1/beta(i) * log(exp(beta(i)*b(i)) + exp(beta(i).*x(:,:,i,:))) - b(i);
    end
    
    if nargout > 1
        for i = 1:nummaps
            tx = squeeze(x(:,:,i,:));
            M = exp(beta(i) * b(i)) + exp(beta(i) * tx);
            da(:,:,i,:) = 1./M .* exp(beta(i) * tx);
            dbeta(:,:,i,:) = -1/(beta(i)^2) .* log(M) + 1./(beta(i)*M) .* (b(i)*exp(beta(i)*b(i)) + tx.*exp(beta(i)*tx));
            db(:,:,i,:) = 1./M .* exp(beta(i)*b(i)) - 1;
        end
    end
end

end