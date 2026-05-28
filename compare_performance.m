%% EVALUATION ON BSDS500 TEST IMAGE
clear; clc; close all;

% 1. LOAD MODELS
% These must be in your current MATLAB directory
load('Traditional_Dict_BSDS.mat'); % Variables: D_learned, lambda
load('Trained_Unrolled_Net.mat');  % Variables: D, K, target_lambda

% 2. DATA PATH - BSDS500
% Update this path to where your BSDS500 test images are stored
test_dir = 'datasets/BSDS300/images/test'; 

if ~exist(test_dir, 'dir')
    error('BSDS500 test directory not found. Please check the path: %s', test_dir);
end

files = dir(fullfile(test_dir, '*.jpg'));
if isempty(files)
    error('No images found in the test directory.');
end

% Pick the first image from the test set
img_name = files(1).name;
img_raw = imread(fullfile(files(1).folder, img_name));

% Pre-process: Grayscale and Double
if size(img_raw, 3) > 1, img_raw = rgb2gray(img_raw); end
img = im2double(img_raw);

% Ensure dimensions are multiples of 8 (patch size)
patch_size = 8;
[H, W] = size(img);
H_new = floor(H/patch_size)*patch_size;
W_new = floor(W/patch_size)*patch_size;
img = img(1:H_new, 1:W_new);

% Extract patches and handle Mean (DC)
Y_test = im2col(img, [patch_size, patch_size], 'distinct');
[m, num_p] = size(Y_test);
p_means = mean(Y_test, 1); % 1 x num_p
Y_norm = Y_test - repmat(p_means, m, 1); % Manual broadcasting

fprintf('Evaluating Image: %s (%d patches)\n', img_name, num_p);
fprintf('------------------------------------------------------------\n');

% ==========================================================
% BLOCK 1: TRADITIONAL IRLS (ITERATIVE)
% ==========================================================
fprintf('Running Traditional IRLS...\n');
X_t = zeros(size(D_learned, 2), num_p);
DtD_t = D_learned' * D_learned;
I_t = eye(size(D_learned, 2));
tic;
for p = 1:num_p
    y = Y_norm(:, p); DtY = D_learned' * y; x = zeros(size(D_learned, 2), 1);
    for iter = 1:50
        xp = x;
        W_diag = (x.^2 + 0.01).^(-0.75); % L1/2 weight
        L = DtD_t + lambda * diag(W_diag) + 1e-4*I_t;
        x = L \ DtY;
        if norm(x-xp)/norm(xp+1e-8) < 1e-3, break; end
    end
    X_t(:, p) = x;
end
time_t = toc;

% Reconstruct separately to avoid dimension errors
Y_rec_t = (D_learned * X_t) + repmat(p_means, m, 1);
img_t = col2im(Y_rec_t, [8 8], [H_new W_new], 'distinct');

% ==========================================================
% BLOCK 2: UNROLLED IRLS-NET (FEEDFORWARD)
% ==========================================================
fprintf('Running Unrolled IRLS-Net (K=%d layers)...\n', K);
X_u = zeros(size(D, 2), num_p);
DtD_u = D' * D;
I_u = eye(size(D, 2));
tic;
for p = 1:num_p
    y = Y_norm(:, p); DtY = D' * y; x = zeros(size(D, 2), 1);
    for k = 1:K
        W_diag = (x.^2 + 0.01).^(-0.75);
        L = DtD_u + target_lambda * diag(W_diag) + 1e-4*I_u;
        x = L \ DtY;
    end
    X_u(:, p) = x;
end
time_u = toc;

% Reconstruct separately
Y_rec_u = (D * X_u) + repmat(p_means, m, 1);
img_u = col2im(Y_rec_u, [8 8], [H_new W_new], 'distinct');

% ==========================================================
% BLOCK 3: FINAL SEPARATE COMPARISON
% ==========================================================
% Calculate metrics for both
psnr_vals = [psnr(img_t, img), psnr(img_u, img)];
ssim_vals = [ssim(img_t, img), ssim(img_u, img)];
spar_vals = [sum(abs(X_t)>0.01,'all')/num_p, sum(abs(X_u)>0.01,'all')/num_p];
time_vals = [time_t, time_u];

% Display output
fprintf('\nHEAD-TO-HEAD COMPARISON:\n');
fprintf('%-20s | %-15s | %-15s\n', 'Metric', 'Traditional', 'Unrolled-Net');
fprintf('------------------------------------------------------------\n');
fprintf('%-20s | %-15.2f | %-15.2f\n', 'PSNR (dB)', psnr_vals(1), psnr_vals(2));
fprintf('%-20s | %-15.4f | %-15.4f\n', 'SSIM', ssim_vals(1), ssim_vals(2));
fprintf('%-20s | %-15.2f | %-15.2f\n', 'Avg Sparsity (L0)', spar_vals(1), spar_vals(2));
fprintf('%-20s | %-15.4f | %-15.4f\n', 'Inference Time (s)', time_vals(1), time_vals(2));

% Visual Comparison
figure('Color', 'w', 'Position', [100 100 1200 400]);
subplot(1,3,1); imshow(img); title('Original Ground Truth');
subplot(1,3,2); imshow(img_t); title(['Traditional (', num2str(psnr_vals(1), '%.2f'), ' dB)']);
subplot(1,3,3); imshow(img_u); title(['Unrolled (', num2str(psnr_vals(2), '%.2f'), ' dB)']);