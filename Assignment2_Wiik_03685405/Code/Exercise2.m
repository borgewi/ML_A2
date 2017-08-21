A = load('A.txt');
B = load('B.txt');
pi = load('pi.txt');
Test = load('Test.txt'); 

N = size(B,1);
[T, repetitions] = size(Test);
gesture1 = zeros(repetitions,1);
gesture2 = zeros(repetitions,1);

for n = 1:repetitions
    o_t = Test(1, n);
    alpha = pi .* B(o_t, :);
    
    for t = 2:T
        o_t = Test(t, n);
        alpha = alpha * A .* B(o_t, :); 
        
    end
    
    logLikelihood = log(sum(alpha));
    if logLikelihood > -120
        gesture1(n) = logLikelihood;
    else
        gesture2(n) = logLikelihood;
    end    
end
