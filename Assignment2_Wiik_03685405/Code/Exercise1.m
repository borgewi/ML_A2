load('dataGMM.mat');
[dim, n] = size(Data);
K = 4;
covariances = zeros(dim,dim,K);
priors = zeros(K);
responsibilities = zeros(size(Data,2),4);

threshold = 0.01;
loglikelihood = threshold + 0.000001;
prev_loglikelihood = 0;

[idx, mu] = kmeans(Data',K);

for k=1:K
   index = idx == k; class = Data(:,index);
   covariances(:,:,k) = cov(class');
   priors(k) = size(class)/size(Data);
end

while loglikelihood - prev_loglikelihood > threshold
    prev_loglikelihood = loglikelihood;
    loglikelihood = 0;
    
    for i = 1:n
        for k= 1:K
            p = mvnpdf(Data(:,i)',mu(k,:),covariances(:,:,k));

            responsibilities(i,k) = (priors(k)*p)/ ... 
            (priors(1)*mvnpdf(Data(:,i)',mu(1,:),covariances(:,:,1)) + ...
            priors(2)*mvnpdf(Data(:,i)',mu(2,:),covariances(:,:,2)) + ...
            priors(3)*mvnpdf(Data(:,i)',mu(3,:),covariances(:,:,3)) + ...
            priors(4)*mvnpdf(Data(:,i)',mu(4,:),covariances(:,:,4)));
        end
    end
    n_k = sum(responsibilities);
    mu  = responsibilities'*Data' ./ repmat(n_k', 1, 2);    
    priors = n_k/n;
    for k=1:K
        zeroMeanData = Data'-mu(k,:);
        covariances(:,:,k) = repmat(responsibilities(:, k), 1, dim)' .*  zeroMeanData' * zeroMeanData / n_k(k);
    end
    %check for convergence
    for i = 1:n
        x_i = Data(:, i);
        x_i = repmat(x_i, 1, k);        
        pdf = mvnpdf(x_i', mu, covariances);
        pdfsum = priors*pdf;
        loglikelihood = loglikelihood + log(pdfsum);
    end
end
