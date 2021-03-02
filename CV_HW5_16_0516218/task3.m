%task3: bag of SIFT + SVM

clc
clear
%run('VLFEATROOT/toolbox/vl_setup')

data_path = 'hw5_data';
categories = {'Bedroom','Coast','Forest','Highway','Industrial','InsideCity','Kitchen' ...
              'LivingRoom','Mountain','Office','OpenCountry','Store','Street','Suburb','TallBuilding'};
num_categories = length(categories);
num_train_per_cat = 100;
num_test_per_cat = 10;
[train_img_paths, test_img_paths, train_labels, test_labels] = ...
img_paths(data_path, categories, num_train_per_cat, num_test_per_cat);

%build vocabularies
vocab_size = 400;
num_samples = 10000;
vocab = build_vocab(train_img_paths, vocab_size, num_samples);
save('vocab.mat', 'vocab');

%use bag of sifts to represent images
train_hists = bags_of_sifts(train_img_paths);
test_hists = bags_of_sifts(test_img_paths);

%linear SVM classifier
num_test_imgs = num_categories*num_test_per_cat;
lambda = 0.0549;
score = zeros(num_categories, num_test_imgs);

%for each category, create temp_label where 1 indicates this category and
%-1 indicates the other categories
%and get the trained classifier for this category
%input the test data then get the scores
for i = 1:num_categories
    temp_label = double(strcmp(train_labels, categories{i}));
    temp_label(find(temp_label==0)) = -1;
    [w, b, info] = vl_svmtrain(train_hists, temp_label, lambda);
    score(i, :) = w'*test_hists + b;
end

%assign the label of test data to the label of 
%the largest score among all the classifiers
[m, idx] = max(score);
predict_labels = cell(num_test_imgs, 1);
for j = 1:num_test_imgs
    predict_labels{j} = categories{idx(j)};
end

%accuracy
match = cellfun(@strcmp, predict_labels, test_labels);
accuracy = sum(match)/(num_test_per_cat*num_categories); %0.6267

end  