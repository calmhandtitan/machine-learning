function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y);

% matrix X has size m*(n+1) 97*2
% matrix theta has size (n+1)*1 2*1
% find hypothesis h(x) = theta0 + theta1*x

predictions = X*theta;	%97*1

% predictions of hypothesis on all m examples
% predictions is a m*1 matrix or a row vector

sqrErrors = (predictions-y).^2;	% do element wise square

J = 1/(2*m) * sum(sqrErrors);

% =========================================================================

end
