function network = train_network(data, y, network, n_epochs, eta)
%UNTITLED Summary of this function goes here
%   Bias is treated like an additional element in the corresponding
%   input/hidden layer

length_input_layer  = network.length_input_layer;
length_hidden_layer = network.length_hidden_layer;
length_output_layer = network.length_output_layer;

no_bias = 0;
loss_function = 'cross_entropy';
if no_bias
    W_layer_1 = 1 - 2*rand(length_hidden_layer, length_input_layer);
    W_layer_2 = 1 - 2*rand(length_output_layer, length_hidden_layer);
else
    W_layer_1 = 1 - 2*rand(length_hidden_layer, length_input_layer + 1);
    W_layer_2 = 1 - 2*rand(length_output_layer, length_hidden_layer + 1);
end
net_layer_1 = zeros(length_hidden_layer, 1);

no_data_points = size(data, 1);
%indices = 1:no_data_points;

for j = 1:n_epochs
    indices = randsample(no_data_points, no_data_points);

    for i = 1:no_data_points
        index = indices(i);
        input = data(index, :)';
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
        output = sigmoid(net_2);
        record_out(index) = output > 0.5;
        
        % backward propagation
        if strcmp(loss_function, 'cross_entropy')
            delta_output = y(index) - output;
            %delta_output = -delta_output;
        else
            delta_output = output.*(1 - output).*(y(index) - output);
        end

        sum_delta_k_w_kj = W_layer_2'.*delta_output;
        delta_hidden = out_layer_1_w_pad.*(1 - out_layer_1_w_pad).*(sum_delta_k_w_kj);
        
        W_layer_1 = W_layer_1 + eta * input_w_pad'       * delta_hidden;
        W_layer_2 = W_layer_2 + eta * out_layer_1_w_pad' * delta_output;
    end
    
    error(j) = mean(abs(y - record_out));
end

plot_error = 1;
if plot_error
    figure()
    plot(error)
    ylim([0 0.5])
    ylabel('Error')
    xlabel('# of epochs')
    grid on
end

network.W_layer_1 = W_layer_1;
network.W_layer_2 = W_layer_2;
network.net_layer_1 = net_layer_1;

end