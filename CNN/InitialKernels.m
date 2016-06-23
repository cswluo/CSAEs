function kernels = InitialKernels(kernelSize,numKernels, numCt)
kernels = zeros(kernelSize^2, numKernels * numCt);
% for i = 1:numKernels
%     kernels(:,i) = random('norm', 0, 0.1, [kernelSize^2,1]);
% end
% kernels = random('norm', 0, 0.1, size(kernels));
fan_in = kernelSize^2 * numCt;
fan_out = kernelSize^2 * numKernels;
% kernels = (rand(size(kernels)) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
kernels = 0.01 * randn(size(kernels));
% invd = find(sqrt(sum(kernels.^2)) > 1);
% if ~isempty(invd)
%     kernels(:,invd) = bsxfun(@rdivide, kernels(:,invd), sqrt(sum(kernels(:,invd).^2)));
% end
end

% kernels = bsxfun(@rdivide, kernels, sqrt(sum(kernels.^2)));