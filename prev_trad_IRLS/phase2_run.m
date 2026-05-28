%% PHASE 2: TRAINING ON BSDS500 IMAGES (WITH TIMING)
clear; clc; close all;

% 1. DATA PATHS
train_dir = 'datasets/BSDS500/images/train'; 
if ~exist(train_dir, 'dir')
    error('BSDS500 training images not found at: %s.', train_dir);
end
files = dir(fullfile(train_dir, '*.jpg'));

% 2. PATCH EXTRACTION PARAMETERS
patch_size = 8;
num_patches = 10000; 
Y = zeros(patch_size^2, num_patches);

fprintf('Extracting %d patches with Contrast Normalization...\n', num_patches);
count = 1;
while count <= num_patches
    img_idx = randi(length(files));
    img = imread(fullfile(files(img_idx).folder, files(img_idx).name));
    if size(img, 3) == 3, img = rgb2gray(img); end
    img = im2double(img);
    
    [h, w] = size(img);
    r = randi(h - patch_size + 1);
    c = randi(w - patch_size + 1);
    patch = img(r:r+patch_size-1, c:c+patch_size-1);
    
    patch_zm = patch(:) - mean(patch(:));
    p_norm = norm(patch_zm);
    
    if p_norm > 1e-4
        Y(:, count) = patch_zm / p_norm; 
        count = count + 1;
    end
end

% 3. L1/2-KSVD DICTIONARY LEARNING (WITH TIMING)
n_atoms = 256; 
lambda = 0.1;   
num_iters = 10; 

fprintf('\nStarting L1/2-KSVD training...\n');
fprintf('Configuration: %d atoms, %d iterations, lambda = %.2f\n', n_atoms, num_iters, lambda);

% --- START TIMER ---
startTime = tic; 

[D_learned, X_learned] = ksvd_l12_train(Y, n_atoms, lambda, num_iters);

% --- STOP TIMER ---
training_time_traditional = toc(startTime); 

% 4. SAVE AND VISUALIZE
% Storing variables for the metric comparison script later
save('Traditional_Dict_BSDS.mat', 'D_learned', 'training_time_traditional', 'lambda', 'num_iters');

fprintf('\n' + repmat('=',1,40) + '\n');
fprintf('TRAINING SUMMARY:\n');
fprintf('Total Training Time: %.2f seconds (%.2f minutes)\n', ...
        training_time_traditional, training_time_traditional/60);
fprintf('Avg Time per Iteration: %.2f seconds\n', training_time_traditional/num_iters);
fprintf('Dictionary saved to: Traditional_Dict_BSDS.mat\n');
fprintf(repmat('=',1,40) + '\n');

% Visualize atoms
figure('Name', 'Traditional IRLS Dictionary Atoms', 'Color', 'w');
for i = 1:64
    subplot(8, 8, i);
    % Contrast stretching for better visualization of features
    atom = reshape(D_learned(:,i), [8, 8]);
    imshow(atom, []);
end
sgtitle('Learned Traditional L_{1/2} Dictionary (BSDS500)');