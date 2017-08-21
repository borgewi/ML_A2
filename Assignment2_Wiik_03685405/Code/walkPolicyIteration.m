function walkPolicyIteration( input_state )
    delta =  [2 4 5 13; 1 3 6 14; 4 2 7 15; 3 1 8 16; 6 8 1 9; 5 7 2 10; 8 6 3 11; 7 5 4 12; 10 12 13 5; 9 11 14 6; 12 10 15 7; 11 9 16 8; 14 16 9 1; 13 15 10 2; 16 14 11 3; 15 13 12 4];
    gamma = 0.99;   
    [n_states, n_actions] = size(delta);       
    policy = ceil(rand(16,1)*4);
    V = zeros(n_states,1);
    rew =[0,-1,0,-1; 0,0,-1,-1; 0,0,-1,-1; 0,-1,0,-1;
        -1,-1,0,0; 0,0,0,0; 0,0,0,0; -1,1,0,0;
        -1,-1,0,0; 0,0,0,0; 0,0,0,0; -1,1,0,0;
        0,-1,0,-1; 0,0,-1,1; 0,0,-1,1; 0,-1,0,-1];

    convergence = false;
    lhs = zeros(n_states);
    rhs = zeros(n_states,1);
    V_given_action = zeros(1,n_actions);
    counter = 0;
    while convergence == false
        counter = counter + 1
        prev_policy = policy;
        
        A = zeros(n_states, n_states);
        for s = 1:n_states
            s_prime = delta(s,policy(s));
            A(s,s_prime) = 1;
            lhs(s,s) = 1;
            lhs(s,s_prime) = -gamma;
            rhs(s) = rew(s,policy(s));
        end
        V = lhs\rhs;
        
        for s = 1:n_states
            for a = 1:n_actions
                V_given_action(a) = rew(s, a) + gamma*V(delta(s,a));
            end
            [~, policy(s)] = max(V_given_action);
        end
        if policy == prev_policy
            convergence = true;
        end
        lhs = zeros(n_states);
        rhs = zeros(n_states,1);
    end

    states = zeros(n_states,1);
    states(1) = input_state;
    for s = 2:n_states
        states(s) = delta(states(s-1), policy(states(s-1)));
    end
    walkshow(states);
end

