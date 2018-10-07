function output = test_network(data, network, y)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
W_layer_1 = network.W_layer_1;
W_layer_2 = network.W_layer_2;

for i = 1:size(data, 1)
        input = data(i, :);
        
        % forward propagation
        net_layer_1 = input*W_layer_1';
        out_layer_1 = sigmoid(net_layer_1);
        net_2 = out_layer_1*W_layer_2';
        output(i) = sigmoid(net_2);
    end
end

