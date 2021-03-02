%return cluster centers as visual words

function vocab = build_vocab(image_paths, vocab_size, num_samples)

%get all feature descriptors
des = [];
for i = 1:length(image_paths) 
    I = imread(image_paths{i});
    I = rescale(I);
    I = single(I);
    [loc, d] = vl_dsift(I, 'fast', 'step', 8);
    des = [des d];
end

%randomly select feature descriptors to do k means clustering
%then return the cluster centers
s = randsample(size(des,2), num_samples);
sample = single(des(:, s));
[centers, ass] = vl_kmeans(sample, vocab_size);
vocab = centers;









