%resize images to 16x16 (ignoring aspect ratio)

function tiny_image = tiny_img(image_paths)

img_size = 16;
tiny_image = zeros(img_size*img_size, length(image_paths));

for i = 1:length(image_paths) 
    I = imread(image_paths{i});

    %resize
    I = imresize(I, [img_size img_size]);
    tiny_image(:, i) = reshape(I.', [1 img_size*img_size]);

    %normalize
    tiny_image(:, i) = rescale(tiny_image(:, i));
end







