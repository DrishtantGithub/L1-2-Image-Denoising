%% HYPERPARAMETER TUNING: UNROLLED IRLS-NET (L1/2 PRIOR)
clear; clc; close all;

load('BSDS500_Patches_Fixed.mat'); % Y_train
% Y_train [64 x N]
[m, N] = size(Y_train);

% Validation Split
val_idx = randperm(N, floor(0.2*N));
train_idx = setdiff(1:N, val_idx);
Y_train_sub = Y_train(:, train_idx);
Y_val = Y_train(:, val_idx);

% 1. DEFINE TUNING RANGE
K_list = 2:1:10;                % Test every depth from 2 to 10
lambda_init_list = [0.01, 0.05, 0.1,0.5,1,5]; 
n = 256; 
epochs_tuning = 8;             

results_mse = zeros(length(K_list), length(lambda_init_list));

fprintf('Starting Exhaustive Tuning for K = 2 to 10...\n');

for i = 1:length(K_list)
    for j = 1:length(lambda_init_list)
        K_curr = K_list(i);
        L_curr = lambda_init_list(j);
        
        fprintf('Testing: K=%d, L=%.2f | ', K_curr, L_curr);
        
        % Initialize
        D = initialize_dct_dict(m, n);
        
        % Train instance
        [D_trained, lambdas_trained] = train_unrolled_instance(Y_train_sub, D, K_curr, L_curr, epochs_tuning);
        
        % Evaluate MSE
        val_mse = compute_val_mse(Y_val, D_trained, K_curr, lambdas_trained);
        results_mse(i,j) = val_mse;
        
        fprintf('Val MSE: %.6f\n', val_mse);
    end
end

% 2. CORRECTED VISUALIZATION
figure('Color', 'w', 'Position', [100 100 900 500]);
hBar = bar(K_list, results_mse);
grid on;
xlabel('Network Depth (K Layers)');
ylabel('Validation MSE');
title('Hyperparameter Tuning: Accuracy vs. Depth');

% Correct Legend Title Handling
lgd = legend(string(lambda_init_list));
title(lgd, 'Init \lambda'); % This is the correct way to set legend titles

% Find the global minimum
[min_val, idx] = min(results_mse(:));
[best_K_idx, best_L_idx] = ind2sub(size(results_mse), idx);
fprintf('\n--- BEST CONFIGURATION ---\n');
fprintf('Optimal K: %d\n', K_list(best_K_idx));
fprintf('Optimal Init Lambda: %.2f\n', lambda_init_list(best_L_idx));
fprintf('Minimum Validation MSE: %.6f\n', min_val);

%% --- KEEP HELPER FUNCTIONS FROM PREVIOUS BLOCK ---
% (initialize_dct_dict, train_unrolled_instance, compute_val_mse)

%% --- HELPER FUNCTIONS ---

function D = initialize_dct_dict(m, n)
    % Creates a structured starting point for the Feedforward atoms
    D = zeros(m, n);
    dct_basis = dctmtx(sqrt(m)); % 8x8 basis
    idx = 1;
    for i = 1:sqrt(m)
        for j = 1:sqrt(m)
            if idx <= n
                atom = dct_basis(i,:)' * dct_basis(j,:);
                D(:, idx) = atom(:);
                idx = idx + 1;
            end
        end
    end
    % Fill remaining atoms with noise
    if n > m
        D(:, m+1:end) = randn(m, n-m) * 0.1;
    end
    D = D ./ sqrt(sum(D.^2, 1) + 1e-8);
end

function [D, lambdas] = train_unrolled_instance(Y_train, D, K, L_start, epochs)
    % Localized manual backprop training
    [m, N] = size(Y_train);
    n = size(D, 2);
    lambdas = L_start * ones(K, 1);
    lr_D = 0.001; lr_L = 0.0001; beta = 1e-4; eps_val = 0.01;
    batchSize = 128;
    
    for ep = 1:epochs
        idx = randperm(N, min(N, 2000)); % Subset for tuning speed
        for i = 1:batchSize:length(idx)-batchSize
            Y = Y_train(:, idx(i:i+batchSize-1));
            % Forward
            x = zeros(n, batchSize);
            DtD = D' * D; DtY = D' * Y;
            for k = 1:K
                L_mat = DtD + lambdas(k)*diag(mean((x.^2 + eps_val).^-0.75, 2)) + beta*eye(n);
                x = L_mat \ DtY;
            end
            % Backward (Simplified)
            res = (D * x) - Y;
            D = D - lr_D * (res * x') / batchSize;
            D = D ./ sqrt(sum(D.^2, 1) + 1e-8);
        end
    end
end

function mse = compute_val_mse(Y, D, K, lambdas)
    % Pure forward pass evaluation
    [m, batchSize] = size(Y);
    n = size(D, 2);
    x = zeros(n, batchSize);
    DtD = D' * D; DtY = D' * Y;
    for k = 1:K
        W = mean((x.^2 + 0.01).^-0.75, 2);
        L_mat = DtD + lambdas(k)*diag(W) + 1e-4*eye(n);
        x = L_mat \ DtY;
    end
    mse = sum((D*x - Y).^2, 'all') / batchSize;
end

function display_dict(D)
    sz = ceil(sqrt(size(D,2)));
    big_img = zeros(sz*9, sz*9);
    for i = 1:size(D,2)
        r = floor((i-1)/sz); c = mod(i-1, sz);
        atom = reshape(D(:,i), [8, 8]);
        atom = (atom - min(atom(:))) / (max(atom(:)) - min(atom(:)) + 1e-8);
        big_img(r*9+1:r*9+8, c*9+1:c*9+8) = atom;
    end
    imagesc(big_img); colormap gray; axis image off;
end