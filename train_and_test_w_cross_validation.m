function [output, mean_error] = train_and_test_w_cross_validation(data_mat, y, indices_set, network, n_epochs, eta)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
n_folds = length(indices_set);
no_data_points = length(y);
for i = 1:n_folds
    test_indices = indices_set{i};
    training_indices{i} = setdiff([1:no_data_points], test_indices);
    data_set = data_mat(training_indices{i}, :);
    y_set    = y(training_indices{i});
    data_test = data_mat(test_indices, :);
    y_test    = y(test_indices);

    network = train_network(data_set, y_set, network, n_epochs, eta);

    [output{i}, error_precent(i)] = test_network(data_test, network, y_test);
    error_num(i) = error_precent(i)*length(y_test)/100;
end

mean_error = sum(error_num)/no_data_points;
end

