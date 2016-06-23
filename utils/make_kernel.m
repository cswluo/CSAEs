% make Gaussian kernel to be used as weighted window in the local
% normalization of the images in the C101.

function k = make_kernel(sz)
% sz is the size of the kernel: sz x sz
% k is the kernel

x = - floor(sz/2) : 1 : floor(sz/2);
s = sz/4; % standard deviation (width) of kernel
k1 = exp(- (x./s).^2);
k = k1' * k1;
k = k ./ sum(k(:));

% plot
figure(1); clf;
subplot(1,2,2)
mesh(x,x,k); 
subplot(1,2,1)
imagesc(k); colormap gray;
axis square

saveas(gcf,['weighted_window' int2str(sz) '.fig'])
save(['weighted_window' int2str(sz) '.mat'],'k')
