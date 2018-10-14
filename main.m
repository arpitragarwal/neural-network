clear; clc; close all;

learning_rate = 0.1;
n_epochs = 100;
n_folds = 10;

filename = 'data/sonar.arff.txt';
[data, metadata] = read_arff_file(filename);
n_features = size(data, 2) - 1;

% Network Specifications
network.length_input_layer = n_features;
network.length_hidden_layer = n_features;
network.length_output_layer = 1;
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
%% Cross Validation Stuff
data_w_label = [data_mat, y'];
% labels are in the last column of the new matrix
% sort the data with respect to the labels
n_attributes = size(data_mat, 2);
no_data_points = size(data_w_label, 1);
data_w_label = sort(data_w_label, n_attributes + 1);

pos_indices_tot = find(y==1);
neg_indices_tot = find(y==0);

no_pos_samples_total = length(pos_indices_tot);
no_neg_samples_total = length(neg_indices_tot);

no_pos_samples_in_set = floor(no_pos_samples_total/n_folds);
no_neg_samples_in_set = floor(no_neg_samples_total/n_folds);

no_pos_samples_rem = no_pos_samples_total;
no_neg_samples_rem = no_neg_samples_total;
for i = 1:n_folds
    if i==n_folds
        sample_pos = 1:no_pos_samples_rem;
        sample_neg = 1:no_neg_samples_rem;
    else
        sample_pos = randsample(no_pos_samples_rem, no_pos_samples_in_set);
        sample_neg = randsample(no_neg_samples_rem, no_neg_samples_in_set);
    end
    indices_pos = pos_indices_tot(sample_pos);
    indices_neg = neg_indices_tot(sample_neg);
    pos_indices_tot = setdiff(pos_indices_tot, indices_pos);
    neg_indices_tot = setdiff(neg_indices_tot, indices_neg);
    no_pos_samples_rem = length(pos_indices_tot);
    no_neg_samples_rem = length(neg_indices_tot);
    indices_set{i} = [indices_neg, indices_pos];
end

% %
% figure()
% for i = 1:n_folds
%     plot(indices_set{i}, i*ones(length(indices_set{i}), 1), 'o')
%     hold on
% end
%%
for i = 1:n_folds
    test_indices = indices_set{i};
    training_indices = setdiff([1:no_data_points], test_indices);
    data_set = data_mat(training_indices, :);
    y_set    = y(training_indices);
    data_test = data_mat(test_indices, :);
    y_test    = y(test_indices);

    network = train_network(data_set, y_set, network, n_epochs, eta);

    [output{i}, error(i)] = test_network(data_test, network, y_test);
end
%%
figure()
plot(error)
hold on
plot([0, n_folds], [mean(error), mean(error)])

% %%
% figure()
% plot(output > 0.5, 'o-', 'LineWidth', 0.1)
% hold on
% plot(y)
% movegui('southeast')