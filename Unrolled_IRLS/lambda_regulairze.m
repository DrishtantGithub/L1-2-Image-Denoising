%% ANALYSIS: SNR vs. LAMBDA
clear; clc;

% 1. Load Data
load('learned_dictionary_bsds500.mat'); 
d_set12 = dir(fullfile('datasets', '**', 'Set12'));
img_clean = im2double(imread(fullfile(d_set12(1).folder, '01.png')));
if size(img_clean, 3) == 3, img_clean = rgb2gray(img_clean); end

% 2. Setup Noise
sigma = 25/255;
img_noisy = img_clean + sigma * randn(size(img_clean));
[h, w] = size(img_clean);
patch_size = 8;

% 3. Lambda Range to Test
lambda_range = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4];
snr_results = zeros(size(lambda_range));
step = 4; % Use a larger step for the parameter sweep to save time

fprintf('Starting Lambda Sweep...\n');

for l_idx = 1:length(lambda_range)
    curr_lambda = lambda_range(l_idx);
    img_out = zeros(h, w);
    weight_mask = zeros(h, w);
    
    for r = 1:step:h-patch_size+1
        for c = 1:step:w-patch_size+1
            patch_noisy = img_noisy(r:r+patch_size-1, c:c+patch_size-1);
            p_mean = mean(patch_noisy(:));
            p_norm = norm(patch_noisy(:) - p_mean);
            
            % Normalize and solve
            p_input = (patch_noisy(:) - p_mean) / (p_norm + eps);
            x = irls_l12_stable(p_input, D_learned, curr_lambda, 15, 1e-4);
            
            % Reconstruct
            p_rec = reshape(D_learned * x, [patch_size, patch_size]) * p_norm + p_mean;
            
            img_out(r:r+patch_size-1, c:c+patch_size-1) = ...
                img_out(r:r+patch_size-1, c:c+patch_size-1) + p_rec;
            weight_mask(r:r+patch_size-1, c:c+patch_size-1) = ...
                weight_mask(r:r+patch_size-1, c:c+patch_size-1) + 1;
        end
    end
    
    img_denoised = img_out ./ (weight_mask + eps);
    img_denoised = max(0, min(1, img_denoised));
    
    % Calculate SNR (Signal-to-Noise Ratio)
    signal_power = mean(img_clean(:).^2);
    noise_power = mean((img_clean(:) - img_denoised(:)).^2);
    snr_results(l_idx) = 10 * log10(signal_power / noise_power);
    
    fprintf('  Lambda: %.2f | SNR: %.2f dB\n', curr_lambda, snr_results(l_idx));
end

% 4. Plotting
figure;
plot(lambda_range, snr_results, '-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('\lambda (Regularization Parameter)');
ylabel('Output SNR (dB)');
title('Denoising Performance vs. \lambda (L_{1/2}-KSVD)');
saveas(gcf, 'media/snr_vs_lambda.png');