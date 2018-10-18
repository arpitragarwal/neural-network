function neural_net(training_filename, num_folds, learning_rate, num_epochs)

%function dt_learn(training_filename, test_filename, m)
%Script for building tree and making predictions

if ~exist('num_epochs','var')
      num_epochs = 100;
end
if ~exist('learning_rate','var')
      learning_rate = 0.1;
end
if ~exist('num_folds','var')
      num_folds = 20;
end
if ~exist('training_filename','var')
      training_filename = 'data/sonar.arff.txt';
end


[data_mat, y, labels, metadata] = read_arff_file(training_filename);
n_features = size(data_mat, 2);

% Network Specifications
network.length_input_layer = n_features;
network.length_hidden_layer = n_features;
network.length_output_layer = 1;
eta = learning_rate;

[indices_set] = generate_cross_validation_data(data_mat, y, num_folds);
[output, ~] = train_and_test_w_cross_validation(data_mat, y, indices_set, network, num_epochs, eta);

print_predictions(output, indices_set, labels, metadata)
end

