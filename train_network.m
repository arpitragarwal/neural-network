function network = train_network(data, y, network, n_epochs, eta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

length_input_layer = network.length_input_layer;
length_hidden_layer = network.length_hidden_layer;
length_output_layer = network.length_output_layer;

W_layer_1 = 1 - 2*rand(length_hidden_layer, length_input_layer);
W_layer_2 = 1 - 2*rand(length_output_layer, length_hidden_layer);
net_layer_1 = zeros(length_hidden_layer, 1);

no_data_points = size(data, 1);
%indices = 1:no_data_points;
indices = randsample(no_data_points, no_data_points);

for j = 1:n_epochs
    for i = 1:no_data_points
        index = indices(i);
        input = data(index, :);
        
        % forward propagation
        net_layer_1 = input * W_layer_1';
        out_layer_1 = sigmoid(net_layer_1);
        net_2 = out_layer_1 * W_layer_2';
        output = sigmoid(net_2);
        record_out(index) = output;
        
        % backward propagation
        delta_output = output.*(1 - output).*(y(index) - output);
        
        sum_delta_k_w_kj = W_layer_2.*delta_output;
        delta_hidden = out_layer_1.*(1 - out_layer_1).*(sum_delta_k_w_kj);
        
        W_layer_1 = W_layer_1 + eta * delta_hidden' * input;
        W_layer_2 = W_layer_2 + eta * delta_output' * out_layer_1;
    end
    
    error(j) = mean(abs(y - record_out));
end
plot_error = 1;
if plot_error
figure()
plot(error)
ylabel('Error')
xlabel('# of epochs')
end
network.W_layer_1 = W_layer_1;
network.W_layer_2 = W_layer_2;
network.net_layer_1 = net_layer_1;

end