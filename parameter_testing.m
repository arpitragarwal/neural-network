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

n_epoch_values = [5, 10, 25, 50, 75, 100];
for i = 1:length(n_epoch_values)
    n_epochs = n_epoch_values(i);
    
    [output, error_b1(i)] = train_and_test_w_cross_validation(data_mat, y, indices_set, network, n_epochs, eta);
    %print_predictions(output, indices_set, labels, metadata)
end

figure()
plot(n_epoch_values, 1-error_b1, 'bo-')
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
    [output, error_b2(i)] = train_and_test_w_cross_validation(data_mat, y, indices_set, network, n_epochs, eta);
end

figure()
plot(n_fold_values, 1-error_b2, 'bo-')
ylim([0.5 1])
ylabel('Accuracy')
xlabel('# folds')
grid on
movegui('southeast')
savefig(gcf, ['../hw3_written_latex/figures/accuracy_vs_n_folds.fig'])

%% PART B.3
n_folds = 10;
n_epochs = 100;

[indices_set] = generate_cross_validation_data(data_mat, y, n_folds);
[output_set, error_b3] = train_and_test_w_cross_validation(data_mat, y, indices_set, network, n_epochs, eta);

indices = [];
outputs = [];
for i = 1:n_folds
    indices = [indices, indices_set{i}];
    outputs = [outputs, output_set{i}];
end
[~, sorted_indices] = sort(indices);
ordered_outputs = outputs(sorted_indices);


clearvars tpr fpr threshold_values
threshold_values = linspace(0, 1, 100);
for j = 1:length(threshold_values)
    threshold = threshold_values(j);
    y_p = ordered_outputs > threshold;
    
    no_false_pos = 0;
    no_true_pos = 0;
    for i = 1:length(y)
        if y(i)==1 && y_p(i)==1
            no_true_pos = no_true_pos + 1;
        elseif y(i)==0 && y_p(i)==1
            no_false_pos = no_false_pos + 1;
        end
    end
    
    no_actual_pos = sum(y);
    tpr(j) = no_true_pos/no_actual_pos;
    fpr(j) = no_false_pos/no_actual_pos;
end

shift = 0.45;
figure()
plot(fpr, tpr, 'o-')
hold on
plot([0, 1], [0, 1], '--')
%plot([0, 1], [0, 1]+shift, '--')
ylim([0 1])
xlim([0 1])
grid on
ylabel('True Positive Rate')
xlabel('False Positive Rate')
savefig(gcf, ['../hw3_written_latex/figures/ROC.fig'])