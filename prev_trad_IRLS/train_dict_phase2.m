%% PHASE 2: TRAINING ON SET12
% 1. DATA PREPARATION
train_dir = '/MATLAB Drive/datasets/DnCNN-master/testsets/Set12'; 
files = dir(fullfile(train_dir, '*.png')); 

if isempty(files)
    error('Still cannot find images. Please check if the extension is .png or .jpg');
end

fprintf('Found %d images. Extracting patches...\n', length(files));

patch_size = 8;
num_patches = 10000; % N = 10,000 patches for training
Y = zeros(patch_size^2, num_patches);

for i = 1:num_patches
    % Pick random image
    img_idx = randi(length(files));
    img = imread(fullfile(files(img_idx).folder, files(img_idx).name));
    
    if size(img, 3) == 3, img = rgb2gray(img); end
    img = im2double(img);
    
    % Pick random patch
    r = randi(size(img,1) - patch_size + 1);
    c = randi(size(img,2) - patch_size + 1);
    patch = img(r:r+patch_size-1, c:c+patch_size-1);
    
    % DC Removal (Essential for Dictionary Learning)
    Y(:, i) = patch(:) - mean(patch(:));
end

% 2. TRAINING PARAMETERS
n_atoms = 256; 
lambda = 0.1;   
num_iters = 10; 

% 3. RUN K-SVD (Ensure ksvd_l12_train.m and irls_l12_stable.m are in your folder)
fprintf('Starting K-SVD training... This may take several minutes.\n');
[D_learned, X_learned] = ksvd_l12_train(Y, n_atoms, lambda, num_iters);

% 4. VISUALIZATION
fprintf('Training complete. Displaying dictionary atoms...\n');
figure;
for i = 1:64 
    subplot(8, 8, i);
    imshow(reshape(D_learned(:,i), [8, 8]), []);
end
sgtitle('Learned L_{1/2} Dictionary Atoms (First 64)');

% Save the learned dictionary for Phase 3 (Denoising)
save('L12_Dictionary.mat', 'D_learned');