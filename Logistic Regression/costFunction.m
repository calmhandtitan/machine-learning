function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

predictions = X*theta;	%(100*3) * (3*1) = 100*1
predictions = sigmoid(predictions);

for i = 1:m;
	J += -y(i)*log(predictions(i)) - (1-y(i))*log(1 - predictions(i));
end;
J = J/m;

for j = 1:size(theta);
	tmp = 0;
	for i = 1:m;
		tmp += ((predictions(i) - y(i)) * X(i, j));
	end;
	grad(j) = tmp/m;
end;

% =============================================================

end
