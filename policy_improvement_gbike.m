function [policy, policy_stable] = policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam)
    [m, n] = size(policy);
    nn = 0:n-1;

    % Poisson probabilities
    P1 = exp(-Lamda(1)) * (Lamda(1) .^ nn) ./ factorial(nn);
    P2 = exp(-Lamda(2)) * (Lamda(2) .^ nn) ./ factorial(nn);
    P3 = exp(-lamda(1)) * (lamda(1) .^ nn) ./ factorial(nn);
    P4 = exp(-lamda(2)) * (lamda(2) .^ nn) ./ factorial(nn);

    policy_stable = true;
    old_policy = policy;

    for i = 1:m
        for j = 1:n
            s1 = i - 1; 
            s2 = j - 1;
            amin = -min(min(s2, m - 1 - s1), 5);
            amax = min(min(s1, n - 1 - s2), 5);

            best_value = -inf;
            best_action = policy(i, j);

            for a = amin:amax
                % Transfer cost (1 free bike from loc1 -> loc2)
                transfer_cost = max(0, abs(a) - (a > 0)) * t;

                % Parking cost
                s1_after = s1 - a;
                s2_after = s2 + a;
                parking_cost = 4 * (s1_after > 10) + 4 * (s2_after > 10);

                R = -transfer_cost - parking_cost;
                Vs_ = 0;

                for n1 = 0:12
                    for n2 = 0:14
                        s1_new = max(0, s1_after - n1);
                        s2_new = max(0, s2_after - n2);
                        reward = (min(n1, s1_after) + min(n2, s2_after)) * r;

                        for n3 = 0:12
                            for n4 = 0:9
                                s1_next = min(s1_new + n3, 20);
                                s2_next = min(s2_new + n4, 20);
                                prob = P1(n1 + 1) * P2(n2 + 1) * P3(n3 + 1) * P4(n4 + 1);

                                Vs_ = Vs_ + prob * V(s1_next + 1, s2_next + 1);
                                R = R + prob * reward;
                            end
                        end
                    end
                end

                if R + gam * Vs_ > best_value
                    best_value = R + gam * Vs_;
                    best_action = a;
                end
            end
            policy(i, j) = best_action;
        end
    end

    if sum(sum(abs(old_policy - policy))) ~= 0
        policy_stable = false;
    end
end