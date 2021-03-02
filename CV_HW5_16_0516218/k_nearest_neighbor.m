%k-nearest neighbor

function predict_labels = k_nearest_neighbor(k, train_data, test_data, train_labels, categories)

n = size(test_data,2);
predict_labels = cell(n, 1);

for i = 1:n
    %find the index of the k nearest neighbors of test data i  
    dist = vl_alldist(train_data,test_data(:, i));
    [d, index] = mink(dist, k);
    
    %assign the label of test data i to the most frequent label of the k nearest neighbors
    vote = zeros(length(categories), 1);
    for l = 1:k
        cat_idx = find(strcmp(categories, train_labels{index(l, 1)}));
        vote(cat_idx, 1) = vote(cat_idx, 1)+1;
    end
    [v, I] = max(vote);
    predict_labels{i} = categories{I};
end