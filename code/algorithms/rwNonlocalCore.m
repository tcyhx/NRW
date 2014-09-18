function prob = rwNonlocalCore(img,seeds,labels,nn,alpha,beta)

%Inputs:    img - input color image
%           seeds - the indices of user labeled pixels
%           labels - input image with user scribbles overlaid
%           nn - the number of nearest neighbors
%           alpha,beta - see equation (3) of the paper
%
%Outputs:   prob - probabilities of each pixel which depth value is 255
%
%
%17/06/03 - HONGXING YUAN

if nargin<4, nn = 10; end;
if nargin<5, alpha = 60; end;
if nargin<6, beta = 1; end;

[X Y Z]=size(img); % image size

% color space conversion
img = colorspace('YDbDr<-', img); % convert color space
disp('get nonlocal laplacian matrix');
L = getLKNN(img,nn,alpha,beta);
boundary = labels(seeds);
disp('solve dirichlet problem with boundary conditions');
prob = dirichletboundaryPCG(L,seeds(:),boundary);
prob = reshape(prob, X, Y);

