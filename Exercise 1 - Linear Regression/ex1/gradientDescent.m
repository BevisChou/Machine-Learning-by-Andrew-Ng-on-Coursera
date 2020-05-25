function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % It's not hard to implement the algorithms.
    
    % tmp0 = theta(1) - alpha * (1 / m) *  sum(X * theta - y);
    % tmp1 = theta(2) - alpha * (1 / m) *  sum((X * theta - y) .* X(:, 2));
    % theta(1) = tmp0;
    % theta(2) = tmp1;
    
    % Please note that there is another way to implement the algorithm.
    
    % tmp0 = theta(1) - alpha * (1 / m) *  sum((X * theta - y) .* X(:, 1));
    % tmp1 = theta(2) - alpha * (1 / m) *  sum((X * theta - y) .* X(:, 2));
    % theta(1) = tmp0;
    % theta(2) = tmp1;
    
    % To further simplify the code, the algoritms could be implemented as
    % the following.
    
    theta = theta - alpha * (1 / m) * X' * (X * theta - y);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
