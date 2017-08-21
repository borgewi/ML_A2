function [ par ] = Exercise1( k )
    load('Data.mat')
    par = {};
    p = 6;
    error_matrix = zeros(p,2);
    Z_Train = [];
    Z_Test = [];
    %Creating folds
    for K = 1:k
        count = 1+(K-1)*(size(Input,2)/k): K*(size(Input,2)/k);
        for i = 1:size(count,2)
            subsets_in(:,i) = Input(:,count(1,i));
            subsets_out(:,i) = Output(:,count(1,i));
        end
        subsets_input{K,1}=subsets_in;
        subsets_output{K,1}=subsets_out;
    end
    %Iterating through every fold for every p-value.
    for P = 1:p
        for fold = 1:k
            X_Test = subsets_input{fold,1};
            Y_Test = subsets_output{fold,1};
            X_Train = [];
            Y_Train = [];
            for train = 1:k
                if train ~= fold
                    X_Train = [X_Train , subsets_input{train,1}];
                    Y_Train = [Y_Train , subsets_output{train,1}];
                end
            end
            %Transpose train and test matrices to allow for consistent matrix multiplication
            X_Train = X_Train';
            Y_Train = Y_Train';
            X_Test = X_Test';
            Y_Test = Y_Test';

            Z_Train = ones(size(X_Train,1),1);
            Z_Test = ones(size(X_Test,1),1);
            %Concatenate new polynomials into z matrix and calculate a-coefficients for this fold
            for iteration = 1:P
                Z_Train = [Z_Train , X_Train(:,1).^iteration , X_Train(:,2).^iteration , (X_Train(:,1).*X_Train(:,2)).^iteration];
                Z_Test = [Z_Test , X_Test(:,1).^iteration , X_Test(:,2).^iteration , (X_Test(:,1).*X_Test(:,2)).^iteration]; 
            end
            A = (inv(Z_Train'*Z_Train)*Z_Train'*Y_Train)';
            %Calculate output with learned parameters
            output_pred = A*Z_Test';
            output_pred = output_pred';
            
            %Calculate position and orientation error for this fold and add errors to error_matrix
            position_error = zeros(size(output_pred,1),1);
            orientation_error = zeros(size(output_pred,1),1);
            for position = 1:size(output_pred,1)
                position_error(position,1) = (sum ( sqrt (( Y_Test(position,1) - output_pred(position,1)).^2 + ...
                                                         (( Y_Test(position,2) - output_pred(position,2))).^2)))/size(output_pred,1);
                orientation_error(position,1) = (sum (sqrt ( ( Y_Test(position,3) - output_pred(position,3)).^2)))/size(output_pred,1);
            end
            error_matrix(P,1) = error_matrix(P,1) + sum(position_error);
            error_matrix(P,2) = error_matrix(P,2) + sum(orientation_error);

        end
    end

    %Find optimal p1 and p2
    [~,p1] = min(error_matrix(:,1));
    [~,p2] = min(error_matrix(:,2));

    %Re-estimating A-matrix for entire dataset with given p1 and p2
    %Input matrix is transposed to allow for constistent matrix multiplications.
    Input = Input';
    %Concatenate new polynomials into z matrices and calculate a-coefficients
    Z_p1 = ones(size(Input,1),1);    
    Z_p2 = ones(size(Input,1),1);   
    for P1 = 1:p1
        Z_p1 = [Z_p1 , Input(:,1).^P1 , Input(:,2).^P1 , (Input(:,1).*Input(:,2)).^P1];
    end
    A_p1 = (inv(Z_p1'*Z_p1)*Z_p1'*Output')';
    for P2 = 1:p2
        Z_p2 = [Z_p2 , Input(:,1).^P2 , Input(:,2).^P2 , (Input(:,2).*Input(:,2)).^P2];
    end
    A_p2 = (inv(Z_p2'*Z_p2)*Z_p2'*Output')';

    par{1,1} = A_p1(1,:)';
    par{1,2} = A_p1(2,:)';
    par{1,3} = A_p2(3,:)';
    save('params','par');
end

