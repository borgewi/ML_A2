function walkQLearning( s )
    init_state = s;

    delta =  [2 4 5 13; 1 3 6 14; 4 2 7 15; 3 1 8 16; 6 8 1 9; 5 7 2 10; 8 6 3 11; 7 5 4 12; 10 12 13 5; 9 11 14 6; 12 10 15 7; 11 9 16 8; 14 16 9 1; 13 15 10 2; 16 14 11 3; 15 13 12 4];
    rew =[0,-1,0,-1; 0,0,-1,-1; 0,0,-1,-1; 0,-1,0,-1;
        -1,-1,0,0; 0,0,0,0; 0,0,0,0; -1,1,0,0;
        -1,-1,0,0; 0,0,0,0; 0,0,0,0; -1,1,0,0;
        0,-1,0,-1; 0,0,-1,1; 0,0,-1,1; 0,-1,0,-1];

    n_states = size(delta,1);
    n_actions = size(delta,2);

    Q = zeros(n_states,n_actions);
    policy = zeros(n_states,1);
    alpha = 0.1;
    epsilon = 20;
    gamma = 0.99;

    T = 11000;
    current_state = init_state;
    while T>0
        [~, opt_action] = max(Q(current_state,:));
        exploration_action = randi([1,4]);
        random_percentage = randi([1,100]);
        if epsilon >= random_percentage
            action = opt_action;
        else 
            action = exploration_action;
        end
        [newstate, reward] = simulateRobot(current_state,action, delta, rew);
        Q(current_state,action) = Q(current_state,action) + alpha*(reward + gamma*max(Q(newstate,:)) - Q(current_state,action));
        current_state = newstate;
        T = T-1;
    end

    for s = 1:n_states
        [~, policy_action] = max(Q(s,:));
        policy(s) = policy_action;
    end

    states = zeros(n_states,1);
    states(1) = init_state;
    for s = 2:n_states
        states(s) = delta(states(s-1), policy(states(s-1)));
    end
    walkshow(states);
end
