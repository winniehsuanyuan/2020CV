%task1: tiny imgs+ nearest neighbor

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

%get tiny images
tiny_test_img = tiny_img(test_img_paths);
tiny_train_img = tiny_img(train_img_paths);

%k-nearest neighbor
k=7;
predict_labels = k_nearest_neighbor(k, tiny_train_img, tiny_test_img, train_labels, categories);

%accuracy
match = cellfun(@strcmp, predict_labels, test_labels);
accuracy = sum(match)/(num_test_per_cat*num_categories) %0.2267
