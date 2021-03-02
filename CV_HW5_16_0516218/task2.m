%task2: bag of SIFT+ nearest neighbor

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

%k-nearest neighbor
k=8; %37
predict_labels = k_nearest_neighbor(k, train_hists, test_hists, train_labels, categories);

%accuracy
match = cellfun(@strcmp, predict_labels, test_labels);
accuracy = sum(match)/(num_test_per_cat*num_categories) %0.56 0.58

