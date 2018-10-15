function [indices_set] = generate_cross_validation_data(data_mat, y, n_folds)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
data_w_label = [data_mat, y'];
% labels are in the last column of the new matrix
% sort the data with respect to the labels
n_attributes = size(data_mat, 2);
no_data_points = size(data_w_label, 1);
[~,idx] = sort(data_w_label(:,end)); % sort just the labels
data_w_label = data_w_label(idx, :);   % sort the whole matrix using the sort indices

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
end

