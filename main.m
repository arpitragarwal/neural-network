clear; clc; close all;

learning_rate = 0.1;
n_epochs = 50;
n_folds = 10;

filename = 'data/sonar.arff.txt';
[data_mat, y, labels, metadata] = read_arff_file(filename);
n_features = size(data_mat, 2);

% Network Specifications
network.length_input_layer = n_features;
network.length_hidden_layer = n_features;
network.length_output_layer = 1;
eta = learning_rate;

%% Cross Validation Stuff
[indices_set] = generate_cross_validation_data(data_mat, y, n_folds);
no_data_points = length(y);
[output, error] = train_and_test_w_cross_validation(data_mat, y, indices_set, network, n_epochs, eta);

print_predictions(output, indices_set, labels, metadata)
%%
% figure()
% plot(error)
% hold on
% plot([0, n_folds], [mean(error), mean(error)])

% %%
% figure()
% plot(output >= 0.5, 'o-', 'LineWidth', 0.1)
% hold on
% plot(y)
% movegui('southeast')