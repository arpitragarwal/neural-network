function [output, error] = test_network(data, network, y)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
W_layer_1 = network.W_layer_1;
W_layer_2 = network.W_layer_2;
threshold = 0.5;

for i = 1:size(data, 1)
    input = data(i, :)';

    no_bias = 0;
    if no_bias
        input_w_pad = input;
    else
        %extra element represents bias
        input_w_pad = [input; 1];
    end
    % forward propagation
    net_layer_1 = (input_w_pad' * W_layer_1')';
    out_layer_1 = sigmoid(net_layer_1);
    if no_bias
        out_layer_1_w_pad = out_layer_1;
    else
        %extra element represents bias
        out_layer_1_w_pad = [out_layer_1; 1];
    end
    net_2 = out_layer_1_w_pad' * W_layer_2';
    output(i) = sigmoid(net_2);
    out_label(i) = output(i) > threshold;
end
error = mean(abs(y - out_label))*100;
end

