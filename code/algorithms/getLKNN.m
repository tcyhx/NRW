function L = getLKNN(img,nn,sigmaC,sigmaS)

[m,n,d] = size(img);
[a b] = ind2sub([m n],1:m*n);
%get nonlocal weights
feature = [reshape(img,m*n,d)';[a;b]/sqrt(m*m+n*n)+rand(2,m*n)*1e-6];
%ind=vl_kdtreequery(vl_kdtreebuild(feature),feature,feature,'NUMNEIGHBORS',nn,'MAXNUMCOMPARISONS',nn*2); %for accuracy
ind=vl_kdtreequery(vl_kdtreebuild(feature),feature,feature,'NUMNEIGHBORS',nn,'MAXNUMCOMPARISONS',nn);
a=reshape(repmat(uint32(1:m*n),nn,1),[],1);
b=reshape(ind,[],1);
row(1:m*n*nn,:)=[min(a,b) max(a,b)];
feature(d+1:d+2,:)=feature(d+1:d+2,:)/100;
clear a b ind;
valDistances  = normalize(sum(abs(feature(1:d,row(:,1))-feature(1:d,row(:,2)))));
geomDistances = normalize(sum(abs(feature(d+1:end,row(:,1))-feature(d+1:end,row(:,2)))));
weightsKNN = exp(-sigmaC*valDistances-sigmaS*geomDistances)+1e-5;
A=sparse(double(row(:,1)),double(row(:,2)),weightsKNN,m*n,m*n); clear weightsKNN
A=A+A';
%get local weights
[~,edges] = lattice(m,n,1);
weights   = makeweights(edges,reshape(img,m*n,d),sigmaC); clear points;
W         = adjacency(edges,weights,m*n); clear edges weights;
A = A + 10*W;
D=spdiags(sum(A,2),0,n*m,n*m); clear value feature row;
L=D-A;