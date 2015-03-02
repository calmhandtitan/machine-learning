function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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



predictions = X*theta;	%(100*3) * (3*1) = 100*1
predictions = sigmoid(predictions);

for i = 1:m;
	J += -y(i)*log(predictions(i)) - (1-y(i))*log(1 - predictions(i));
end;
J = J/m;


tmp = (lambda* (sum(theta.^2) - theta(1).^2));	%should not regularize theta(1)
J += tmp/(2*m);


tmp = 0;
for i = 1:m;
	tmp += (predictions(i) - y(i)) * X(i, 1);	
end;
grad(1) = tmp/m;

for j = 2:size(theta);
	tmp = 0;
	for i = 1:m;
		tmp += (predictions(i) - y(i)) * X(i, j);
	end;
	grad(j) = tmp/m + (lambda * theta(j))/m;
end;



% =============================================================

end
