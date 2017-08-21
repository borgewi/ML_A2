function [ newstate, reward ] = simulateRobot( state, action, delta, reward_matrix )
    newstate = delta(state,action);
    reward  = reward_matrix(state,action);
end

