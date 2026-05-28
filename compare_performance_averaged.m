%% TARGETED EVALUATION: IMAGE 16068.jpg
clear; clc; close all;

% 1. LOAD MODELS
load('Traditional_Dict_BSDS.mat'); % D_learned, lambda
load('Trained_Unrolled_Net.mat');  % D, K, target_lambda

% 2. LOAD SPECIFIC IMAGE
img_path = 'datasets/BSDS500/images/test/130014.jpg'; 
if ~exist(img_path, 'file')
    error('Image 16068.jpg not found. Please verify the path.');
end

img_raw = imread(img_path);
if size(img_raw, 3) > 1, img_raw = rgb2gray(img_raw); end
img = im2double(img_raw);

% Pre-process (8x8 patches)
patch_size = 8;
[H, W] = size(img);
H_new = floor(H/patch_size)*patch_size;
W_new = floor(W/patch_size)*patch_size;
img = img(1:H_new, 1:W_new);

Y_test = im2col(img, [patch_size, patch_size], 'distinct');
[m, num_p] = size(Y_test);
p_means = mean(Y_test, 1);
Y_norm = Y_test - repmat(p_means, m, 1);

% --- BLOCK 1: TRADITIONAL IRLS ---
X_t = zeros(size(D_learned, 2), num_p);
DtD_t = D_learned' * D_learned;
I_t = eye(size(D_learned, 2));
tic;
for p = 1:num_p
    y = Y_norm(:, p); DtY = D_learned' * y; x = zeros(size(D_learned, 2), 1);
    for iter = 1:50
        xp = x;
        W_diag = (x.^2 + 0.01).^(-0.75);
        L = DtD_t + lambda * diag(W_diag) + 1e-4*I_t;
        x = L \ DtY;
        if norm(x-xp)/norm(xp+1e-8) < 1e-3, break; end
    end
    X_t(:, p) = x;
end
t_t = toc;
img_t = col2im((D_learned * X_t) + repmat(p_means, m, 1), [8 8], [H_new W_new], 'distinct');

% --- BLOCK 2: UNROLLED IRLS-NET ---
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
t_u = toc;
img_u = col2im((D * X_u) + repmat(p_means, m, 1), [8 8], [H_new W_new], 'distinct');

% --- 3. RESULTS CALCULATION ---
metrics = [psnr(img_t, img), psnr(img_u, img); ...
           ssim(img_t, img), ssim(img_u, img); ...
           sum(abs(X_t)>0.01,'all')/num_p, sum(abs(X_u)>0.01,'all')/num_p; ...
           t_t, t_u];

% --- 4. DISPLAY RESULTS ---
fprintf('\nRESULTS FOR IMAGE 16068.jpg\n');
fprintf('------------------------------------------------------------\n');
fprintf('%-20s | %-15s | %-15s\n', 'Metric', 'Traditional', 'Unrolled-Net');
fprintf('------------------------------------------------------------\n');
fprintf('%-20s | %-15.2f | %-15.2f\n', 'PSNR (dB)', metrics(1,1), metrics(1,2));
fprintf('%-20s | %-15.4f | %-15.4f\n', 'SSIM', metrics(2,1), metrics(2,2));
fprintf('%-20s | %-15.2f | %-15.2f\n', 'Sparsity (L0)', metrics(3,1), metrics(3,2));
fprintf('%-20s | %-15.4f | %-15.4f\n', 'Time (s)', metrics(4,1), metrics(4,2));
fprintf('------------------------------------------------------------\n');

% --- 5. VISUALIZATION ---
figure('Color', 'w', 'Position', [100, 100, 1000, 700]);

subplot(2,3,1); imshow(img); title('Original Ground Truth');
subplot(2,3,2); imshow(img_t); title(sprintf('Traditional (%.2f dB)', metrics(1,1)));
subplot(2,3,3); imshow(img_u); title(sprintf('Unrolled (%.2f dB)', metrics(1,2)));

% Error Maps (multiplied by 5 for visibility)
subplot(2,3,5); imshow(abs(img - img_t)*5, []); title('Trad. Error (x5)');
subplot(2,3,6); imshow(abs(img - img_u)*5, []); title('Unroll. Error (x5)');