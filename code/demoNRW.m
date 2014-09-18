%Semi-automatic 2d-to-3d conversion using nonlocal random walks
%04/27/14 - Hongxing Yuan
%
%The code  has been tested in MATLAB R2013a running on Windows 7(64bit). 
%- Note:
%(1) If you plan to use this software or any variant of it, please, refer the following paper:
%    H. X. Yuan, S. Q. Wu, Nonlocal Random Walks for Semi-Automatic 2D-to-3D Image Conversion, 
%    IEEE Transactions on Signal Processing
%(2) We use vlfeat to perform KNN computation, so you must set up vlfeat toolbox, which can be downloaded from:
%    http://www.vlfeat.org/
%(3) For any questions, suggestions or bug reports, please, contact yuanhx@mail.ustc.edu.cn

clear all;
close all;
addpath('algorithms');
run('3rd/vlfeat/toolbox/vl_setup'); %add the path of vlfeat

%% parameters
dirname  = '../datasets/Philips';
filename = 'image00420';
savename = 'depth_NRW';
nn       = 20;    %KNN neighbors number
alpha    = 60;    %The free paremeter controling how color dissimilarity
beta     = 6000;  %The free paremeter controling how far between neighboring pixels

%Read image
img   = double(imread(sprintf('%s/%s.png',dirname,filename)))/255; %read input image
lbls  = double(imread(sprintf('%s/strokes.png',dirname)))/255;     %read user labeled image
mask  = imread(sprintf('%s/mask.png',dirname));
mask  = im2bw(mask,0);
seeds = find(mask==1);
%% main routine
prob = rwNonlocalCore(img,seeds,lbls,nn,alpha,beta);
depth = uint8(prob.*255);                                      %mapping probabilites to depth values
imshow(depth);
imwrite(depth,sprintf('%s/%s.png',dirname,savename));          %save results