function print_predictions(output, indices_set, labels, metadata)
%UNTITLED Summary of this function goes here

n_folds = length(indices_set);
no_data_points = length(labels);
% Combine and sort the outputs
all_outputs = [];
all_indices = [];
fold_number = [];
for i = 1:n_folds
    all_outputs = [all_outputs, output{i}];
    all_indices = [all_indices, indices_set{i}];
    fold_number = [fold_number, i*ones(1, length(output{i}))];
end
tmp = [all_indices', all_outputs', fold_number'];
[~, idx] = sort(all_indices');
sorted_outputs = tmp(idx, :);

% print
sorted_output_values = sorted_outputs(:, 2);
fold_number = sorted_outputs(:, 3);
for i = 1:no_data_points
    if sorted_output_values(i) < 0.5
        predicted_class = metadata.attribute_values{end}{1};
        confidence(i) = 1 - sorted_output_values(i);
    else
        predicted_class = metadata.attribute_values{end}{2};
        confidence(i) = sorted_output_values(i);
    end
    actual_class = labels{i};
    display([num2str(fold_number(i)-1), ' ', predicted_class, ' ', actual_class, ' ', num2str(confidence(i))]);
end
end

