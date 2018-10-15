clear; clc; close all;

learning_rate = 0.1;
filename = 'data/sonar.arff.txt';
[data_mat, y, labels, metadata] = read_arff_file(filename);
n_features = size(data_mat, 2);
no_data_points = length(y);

% Network Specifications
network.length_input_layer = n_features;
network.length_hidden_layer = n_features;
network.length_output_layer = 1;
eta = learning_rate;

%% PART B.1
n_folds = 10;
[indices_set] = generate_cross_validation_data(data_mat, y, n_folds);

n_epoch_values = [25, 50, 75, 100];
for i = 1:length(n_epoch_values)
    n_epochs = n_epoch_values(i);

    [output, error_b(i)] = train_and_test_w_cross_validation(data_mat, y, indices_set, network, n_epochs, eta);
    %print_predictions(output, indices_set, labels, metadata)
end

figure()
plot(n_epoch_values, 1-error_b, 'bo-')
ylim([0.5 1])
ylabel('Accuracy')
xlabel('# epochs')
grid on
movegui('northeast')
savefig(gcf, ['../hw3_written_latex/figures/accuracy_vs_n_epochs.fig'])
%% PART B.2
n_fold_values = [5, 10, 15, 20, 25];
n_epochs = 50;

for i = 1:length(n_fold_values)
    n_folds = n_fold_values(i);
    [indices_set] = generate_cross_validation_data(data_mat, y, n_folds);
    [output, error_c(i)] = train_and_test_w_cross_validation(data_mat, y, indices_set, network, n_epochs, eta);
end

figure()
plot(n_fold_values, 1-error_c, 'bo-')
ylim([0.5 1])
ylabel('Accuracy')
xlabel('# folds')
grid on
movegui('southeast')
savefig(gcf, ['../hw3_written_latex/figures/accuracy_vs_n_folds.fig'])