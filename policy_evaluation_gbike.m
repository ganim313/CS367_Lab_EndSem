function [V] = policy_evaluation_gbike(policy, Lamda, lamda, r, t, gam)
    [m, n] = size(policy);
    nn = 0:n-1;

    % Poisson probabilities
    P1 = exp(-Lamda(1)) * (Lamda(1) .^ nn) ./ factorial(nn);
    P2 = exp(-Lamda(2)) * (Lamda(2) .^ nn) ./ factorial(nn);
    P3 = exp(-lamda(1)) * (lamda(1) .^ nn) ./ factorial(nn);
    P4 = exp(-lamda(2)) * (lamda(2) .^ nn) ./ factorial(nn);

    theta = 0.1;
    delta = inf;
    V = zeros(m, n);

    while delta > theta
        v = V;
        for i = 1:m
            for j = 1:n
                s1 = i - 1; 
                s2 = j - 1; 
                a = policy(i, j);

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
                V(i, j) = R + gam * Vs_;
            end
        end
        delta = max(max(abs(V - v)));
    end
end