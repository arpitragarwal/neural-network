clear; clc; close all;

filename = 'data/sonar.arff.txt';
[data, metadata] = read_arff_file(filename);
n_features = size(data, 2) - 1;

learning_rate = 0.1;
n_epochs = 100;
eta = learning_rate;

tmp = data(:, 1:end - 1);
for i = 1:size(tmp, 1)
    for j = 1:size(tmp, 2)
        data_mat(i, j) = str2num(tmp{i, j});
    end
end
labels = data(:, end);
class_names = metadata.attribute_values{end};
for i = 1:length(labels)
    if strcmp(labels{i}, class_names{2})
        y(i) = 1;
    end
end
%%
% Network Specifications
network.length_input_layer = n_features;
network.length_hidden_layer = n_features;
network.length_output_layer = 1;

network = train_network(data_mat, y, network, n_epochs, eta);

output = test_network(data_mat, network, y);
%%
figure()
plot(output)
hold on
plot(y)