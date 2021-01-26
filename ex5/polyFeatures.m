function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

% Initialize X_poly
X_poly = zeros(numel(X), p);
X_poly(:,1) = X;

% Set each column to the some power of X
for pow = 1:p
  X_poly(:,pow) = X .^ pow;

end
