function image = show_centroids(centroids, H, W)
  if (nargin < 3)
    W = H;
  end
  N=size(centroids,2)/(H*W);
  assert(N == 3 || N == 1);  % color and gray images
  
  K=size(centroids,1);
  COLS=round(sqrt(K));
  ROWS=ceil(K / COLS);
  COUNT=COLS * ROWS;
  MAR = 1;
  clf; hold on;
  image=ones(ROWS*(H+MAR), COLS*(W+MAR), N)*100;
  for i=1:size(centroids,1)
    r= floor((i-1) / COLS);
    c= mod(i-1, COLS);
    image((r*(H+MAR)+1):((r+1)*(H+MAR))-MAR,(c*(W+MAR)+1):((c+1)*(W+MAR))-MAR,:) = reshape(centroids(i,1:W*H*N),H,W,N);
  end

  mn=-1.5;
  mx=+1.5;
  image = (image - mn) / (mx - mn);
  imshow(image);
