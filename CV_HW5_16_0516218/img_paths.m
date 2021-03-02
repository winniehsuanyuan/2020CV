%preparing image paths

function [train_img_paths, test_img_paths, train_labels, test_labels] = ... 
    img_paths(data_path, categories, num_train_per_cat, num_test_per_cat)

num_categories = length(categories); 
train_img_paths = cell(num_categories * num_train_per_cat, 1);
test_img_paths  = cell(num_categories * num_test_per_cat, 1);
train_labels = cell(num_categories * num_train_per_cat, 1);
test_labels  = cell(num_categories * num_test_per_cat, 1);

for i=1:num_categories
   catdirname = lower(categories{i});
   images = dir( fullfile(data_path, 'train', catdirname, '*.jpg'));
   for j=1:num_train_per_cat
       train_img_paths{(i-1)*num_train_per_cat + j} = fullfile(data_path, 'train', catdirname, images(j).name);
       train_labels{(i-1)*num_train_per_cat + j} = categories{i};
   end
   
   images = dir( fullfile(data_path, 'test', catdirname, '*.jpg'));
   for j=1:num_test_per_cat
       test_img_paths{(i-1)*num_test_per_cat + j} = fullfile(data_path, 'test', catdirname, images(j).name);
       test_labels{(i-1)*num_test_per_cat + j} = categories{i};
   end
end