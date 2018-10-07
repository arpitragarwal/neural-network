function [data, metadata] = read_arff_file(filename)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

fid = fopen(filename);
text = textscan(fid, '%s');

n_attributes = 0;
data_start_index = 0;
is_attribute_numeric = 0;
for i = 1:length(text{1,1})
    if (strcmp(text{1,1}{i},'@attribute'))
        n_attributes = n_attributes + 1;
    elseif (strcmp(text{1,1}{i},'@data'))
        data_start_index = i + 1;
    end
    
    if (strcmp(text{1,1}{i},'real') || strcmp(text{1,1}{i},'numeric'))
        is_attribute_numeric(n_attributes) = 1;
    end
end


% reading the header information here
fid2 = fopen(filename);
tline = fgetl(fid2);
for i = 1:n_attributes + 1
    tmp{i, 1} = tline;
    tline = fgetl(fid2);
end
fclose(fid2);

for i = 1:n_attributes+1
    if i ==1
        tmp2{i} = regexp(tmp{i}, " ", 'split');
        dataset_name = tmp2{i}{2};
    elseif i > 1
        tmp2{i} = regexp(tmp{i}, "'", 'split');
        attribute_names{i - 1} = tmp2{i}{2};
        attribute_values{i - 1} = regexp(tmp2{i}{3}, ",", 'split');
        attribute_values{i - 1} = replace(attribute_values{i - 1}, '{', '');
        attribute_values{i - 1} = replace(attribute_values{i - 1}, '}', '');
        attribute_values{i - 1} = replace(attribute_values{i - 1}, ' ', '');
    end
end


is_attribute_numeric = [is_attribute_numeric, zeros(1, n_attributes - length(is_attribute_numeric))];

data_raw = text{1,1}(data_start_index:end);

for i = 1:length(data_raw)
    tmp{i, :} = strsplit(data_raw{i},',');
end

for i = 1:length(data_raw)
    for j = 1:n_attributes
        data{i, j} = tmp{i}{j};
    end
end

metadata.is_attribute_numeric = is_attribute_numeric;
metadata.attribute_names = attribute_names;
metadata.attribute_values = attribute_values;
metadata.dataset_name = dataset_name;
fclose(fid);
end