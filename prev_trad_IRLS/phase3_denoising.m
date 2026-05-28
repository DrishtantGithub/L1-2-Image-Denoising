%% PHASE 3: IMAGE DENOISING (Optimized)
clear; clc;

% 1. Load Dictionary and Identify Paths
load('learned_dictionary_bsds500.mat'); % Ensure this file exists from Phase 2
d_set12 = dir(fullfile('datasets', '**', 'Set12'));
if isempty(d_set12), error('Set12 folder not found.'); end

% Select a test image (e.g., '01.png' or '06.png')
test_img_name = '01.png'; 
test_img_path = fullfile(d_set12(1).folder, test_img_name);

img_clean = im2double(imread(test_img_path));
if size(img_clean, 3) == 3, img_clean = rgb2gray(img_clean); end

% 2. Add Gaussian Noise (sigma=25 is standard)
sigma = 25/255; 
img_noisy = img_clean + sigma * randn(size(img_clean));

% 3. Parameters
[h, w] = size(img_clean);
patch_size = 8;
img_out = zeros(h, w);
weight_mask = zeros(h, w);

% Tune lambda based on noise level: higher sigma usually needs slightly higher lambda
lambda_denoise = 0.15; 
step = 2; % Set to 1 for best quality, 2 or 3 for significantly faster execution

% 4. Sliding Window Denoising
fprintf('Denoising %s (Step size: %d)...\n', test_img_name, step);
tic; % Start timer

for r = 1:step:h-patch_size+1
    for c = 1:step:w-patch_size+1
        % Extract noisy patch
        patch_noisy = img_noisy(r:r+patch_size-1, c:c+patch_size-1);
        p_mean = mean(patch_noisy(:));
        patch_normalized = patch_noisy(:) - p_mean;
        
        % Contrast Normalization for the solver (matching Phase 2 training)
        p_norm = norm(patch_normalized);
        if p_norm > 1e-4
            p_input = patch_normalized / p_norm;
        else
            p_input = patch_normalized;
        end
        
        % Sparse Coding with Stability Ridge Term
        % Using fewer iterations (15) for inference speed
        x = irls_l12_stable(p_input, D_learned, lambda_denoise, 15, 1e-4);
        
        % Reconstruct and re-scale
        patch_rec = reshape(D_learned * x, [patch_size, patch_size]);
        if p_norm > 1e-4, patch_rec = patch_rec * p_norm; end
        patch_rec = patch_rec + p_mean;
        
        % Accumulate results
        img_out(r:r+patch_size-1, c:c+patch_size-1) = ...
            img_out(r:r+patch_size-1, c:c+patch_size-1) + patch_rec;
        weight_mask(r:r+patch_size-1, c:c+patch_size-1) = ...
            weight_mask(r:r+patch_size-1, c:c+patch_size-1) + 1;
    end
end
toc;

% 5. Final Averaging and Clipping
img_denoised = img_out ./ (weight_mask + eps); 
img_denoised(weight_mask == 0) = img_noisy(weight_mask == 0); % Handle boundaries
img_denoised = max(0, min(1, img_denoised)); % Clip to [0, 1] range

% 6. Performance Evaluation
psnr_noisy = psnr(img_noisy, img_clean);
psnr_denoised = psnr(img_denoised, img_clean);

% 7. Display Results
figure('Name', ['Denoising Result: ', test_img_name]);
subplot(1,3,1); imshow(img_clean); title('Original');
subplot(1,3,2); imshow(img_noisy); title(['Noisy (', num2str(psnr_noisy, '%.2f'), ' dB)']);
subplot(1,3,3); imshow(img_denoised); title(['Denoised (', num2str(psnr_denoised, '%.2f'), ' dB)']);

fprintf('Denoising Complete.\nPSNR Improvement: %.2f dB\n', psnr_denoised - psnr_noisy);