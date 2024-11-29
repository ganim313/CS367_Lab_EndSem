clear all;
close all;

Lamda = [3 4]; % Rental request arrival
lamda = [3 2]; % Return request arrival

r = 10; % 10 rupee rental reward
t = 2; % 2 rupee transfer fees

policy = zeros(21, 21); % Initial policy of no transfer, transfer policy(i, j) from location 1 to location 2
gam = 0.9;

policystable = false;
count = 0;

% Iterative process for policy evaluation and improvement
while ~policystable
    % Policy Evaluation Step
    V = policy_evaluation_gbike(policy, Lamda, lamda, r, t, gam);
    
    % Policy Improvement Step
    [policy, policystable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam);
    
    count = count + 1;
    % Optional: Visualize during each iteration (for debugging or monitoring)
    % figure(1);
    % subplot(2,1,1);contour(policy, [-5:5]);
    % subplot(2,1,2);surf(V);
    % pause();
end

% Final visualization of policy and value function
figure(1); 
subplot(2, 1, 1); contour(policy, [-5:5]);
subplot(2, 1, 2); surf(V);