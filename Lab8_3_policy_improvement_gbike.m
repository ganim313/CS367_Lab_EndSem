function [policy, policy_stable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam)

[m, n] = size(policy);
nn = 0:n-1;

% Calculate the probability distributions for arrivals and returns
P1 = exp(-Lamda(1)) * (Lamda(1).^nn) ./ factorial(nn); 
P2 = exp(-Lamda(2)) * (Lamda(2).^nn) ./ factorial(nn); 
P3 = exp(-lamda(1)) * (lamda(1).^nn) ./ factorial(nn); 
P4 = exp(-lamda(2)) * (lamda(2).^nn) ./ factorial(nn);

policy_stable = true;
old_policy = policy;

% Policy Improvement loop
for i = 1:m
    for j = 1:n
        s1 = i - 1; 
        s2 = j - 1; % state (0-20, 0-20)
        amin = -min(min(s2, m - 1 - s1), 5);
        amax = min(min(s1, n - 1 - s2), 5);
        v_ = -inf; 
        
        % Loop through all possible actions and evaluate
        for a = amin:amax
            R = -abs(a) * t; % Reward for taking action a
            Vs_ = 0;
            s1_ = s1 - a; 
            s2_ = s2 + a;
            
            % Loop through possible arrivals and returns
            for n1 = 0:12
                for n2 = 0:14
                    s1__ = s1_ - min(n1, s1_);
                    s2__ = s2_ - min(n2, s2_);
                    for n3 = 0:12
                        for n4 = 0:9
                            s1___ = s1__ + min(n3, 20 - s1__);
                            s2___ = s2__ + min(n4, 20 - s2__);
                            Vs_ = Vs_ + P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * V(s1___ + 1, s2___ + 1);
                            R = R + (P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1) * (min(n1, s1_) + min(n2, s2_))) * r;
                        end
                    end
                end
            end
            % Update policy based on max value
            if (R + gam * Vs_ > v_)
                v_ = R + gam * Vs_; 
                policy(i, j) = a;
            end
        end
    end
end

% Check if the policy has changed
if sum(sum(abs(old_policy - policy))) ~= 0
    policy_stable = false;
end
end
