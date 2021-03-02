%represent image features as histograms

function hist = bags_of_sifts(image_paths)

load('vocab.mat');
vocab_size = size(vocab, 2); 
n = length(image_paths);
hist = zeros(vocab_size, n);

%for each descriptor find the closest visual word
%build a histogram where the horizontal axis is the visual words and 
%the vertical axis is the number of features assigned to each visual word
for i = 1:n
    I = imread(image_paths{i});
    I = rescale(I);
    I = single(I);
    [loc, d] = vl_dsift(I, 'fast', 'step', 8);
    d = single(d);
    dist = vl_alldist2(vocab, d);
    [m, index] = min(dist);  
    hist(index, i) = hist(index, i)+1;
end

%normalize the histograms so that they are invariant to the size of imgs
hist = normalize(hist);


    




