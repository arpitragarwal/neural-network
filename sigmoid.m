function [output] = sigmoid(net)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    output = 1./(1 + exp(-net));
end

