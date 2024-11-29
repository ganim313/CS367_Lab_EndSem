% Gbike Bicycle Rental Problem - Policy Iteration with Additional Constraints
% Updated for free shuttle and parking space cost
clear all;
close all;

% Parameters
Lamda = [3 4]; % Rental request arrival rates
lamda = [3 2]; % Return rates
r = 10; % INR 10 per rental reward
t = 2; % INR 2 transfer fee per bike
gam = 0.9; % Discount factor

% Initialize policy and value function
policy = zeros(21, 21); % No transfer initially
V = zeros(21, 21); % Initial value function
policy_stable = false;
iteration_count = 0;

% Policy Iteration
while ~policy_stable
    % Policy Evaluation
    V = policy_evaluation_gbike(policy, Lamda, lamda, r, t, gam);
    
    % Policy Improvement
    [policy, policy_stable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam);
    
    iteration_count = iteration_count + 1;
    fprintf('Iteration %d complete.\n', iteration_count);
end

% Visualization
figure;
subplot(2, 1, 1);
contour(policy, -5:5);
title('Optimal Policy');
xlabel('Location 2 Bikes');
ylabel('Location 1 Bikes');

subplot(2, 1, 2);
surf(V);
title('Value Function');
xlabel('Location 2 Bikes');
ylabel('Location 1 Bikes');
zlabel('Value');
