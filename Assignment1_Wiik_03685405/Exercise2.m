function [ d_opt, error_min, CM ] = Exercise2( d_max )
training_images = loadMNISTImages('train-images.idx3-ubyte');
training_labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
%Find zero mean image
meanImage = mean(training_images,2);
zeroMeanImages = training_images - meanImage;
zeroMeanTestImages = test_images - meanImage;

S = cov(zeroMeanImages'); 
[eigVecs, ~] = eig(S);
%error_min is initially declared to 100%, which is the maximum error. 
%d_opt is initially declared to -1, which is a value it can't obtain in the for loop.
error_min = 100;
d_opt = -1;
%Preallocating matrices for speed improvements
predicted_labels = zeros(size(test_labels));
predicted_labels_ideal = zeros(size(test_labels));
errorVector = zeros(d_max,1);
%Looping through all values of d
for d = 1:d_max
    %The transformation matrix is initialized with zeros and assigned the d
    %eigenvectors corresponding to the d largest eigenvectors.
    W = zeros(d,length(eigVecs));
    for i = 1:d
        W(i,:) = eigVecs(:,end-i+1);
    end
    %Transforming training set
    x_train = W*zeroMeanImages;
    % Splitting data into classes
    index0 = training_labels == 0; class0 = x_train(:,index0);
    index1 = training_labels == 1; class1 = x_train(:,index1);
    index2 = training_labels == 2; class2 = x_train(:,index2);
    index3 = training_labels == 3; class3 = x_train(:,index3);
    index4 = training_labels == 4; class4 = x_train(:,index4);
    index5 = training_labels == 5; class5 = x_train(:,index5);
    index6 = training_labels == 6; class6 = x_train(:,index6);
    index7 = training_labels == 7; class7 = x_train(:,index7);
    index8 = training_labels == 8; class8 = x_train(:,index8);
    index9 = training_labels == 9; class9 = x_train(:,index9);
    % Mean and covariance of each digit class
    mu_c0 = mean(class0,2);
    mu_c1 = mean(class1,2);
    mu_c2 = mean(class2,2);
    mu_c3 = mean(class3,2);
    mu_c4 = mean(class4,2);
    mu_c5 = mean(class5,2);
    mu_c6 = mean(class6,2);
    mu_c7 = mean(class7,2);
    mu_c8 = mean(class8,2);
    mu_c9 = mean(class9,2);
    
    cov_c0 = cov(class0');
    cov_c1 = cov(class1');
    cov_c2 = cov(class2');
    cov_c3 = cov(class3');
    cov_c4 = cov(class4');
    cov_c5 = cov(class5');
    cov_c6 = cov(class6');
    cov_c7 = cov(class7');
    cov_c8 = cov(class8');
    cov_c9 = cov(class9');
    %Project learned eigenbasis onto zero meaned test images
    x = W*zeroMeanTestImages;
    %Find likelihoods for each class and likelihood matrix
    likelihoodMatrix = zeros(10, size(x,2));
    for el = 1:size(test_images,2)
        L0 = mvnpdf(x(:,el),mu_c0,cov_c0);
        L1 = mvnpdf(x(:,el),mu_c1,cov_c1);
        L2 = mvnpdf(x(:,el),mu_c2,cov_c2);
        L3 = mvnpdf(x(:,el),mu_c3,cov_c3);
        L4 = mvnpdf(x(:,el),mu_c4,cov_c4);
        L5 = mvnpdf(x(:,el),mu_c5,cov_c5);
        L6 = mvnpdf(x(:,el),mu_c6,cov_c6);
        L7 = mvnpdf(x(:,el),mu_c7,cov_c7);
        L8 = mvnpdf(x(:,el),mu_c8,cov_c8);
        L9 = mvnpdf(x(:,el),mu_c9,cov_c9);
        likelihoodMatrix(:,el) = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9];
    end
    %Find number of wrongly predicted images and use it to calculate error.
    WrongImage = 0;
    for image = 1:size(test_labels(:,1))
        [~, I] = max(likelihoodMatrix(:,image));
        predicted_labels(image) = I - 1;
        if(predicted_labels(image) ~= test_labels(image))
            WrongImage = WrongImage + 1;
        end
    end
    errorVector(d) = (WrongImage/size(test_images,2))*100;     
    if(errorVector(d) < error_min)
       error_min = errorVector(d);
       d_opt = d;
       predicted_labels_ideal = predicted_labels; 
    end
end

figure(1);
plot(errorVector)
title('Image Classification Errors')
xlabel('Number of Principal Components')
ylabel('Error percentage')

CM = confusionmat(test_labels,predicted_labels_ideal);
helperDisplayConfusionMatrix(CM)


end

