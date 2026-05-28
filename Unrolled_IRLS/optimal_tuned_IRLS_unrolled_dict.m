%% FINAL TRAINING: UNROLLED IRLS-NET (K=5, L=0.05)
clear; clc; close all;

% --- 1. DATA LOADING ---
load('BSDS500_Patches_Fixed.mat'); % Y_train % Y_train [64 x N]
[m, N] = size(Y_train);
n = 256; 

% --- 2. OPTIMIZED HYPERPARAMETERS ---
K = 5;           % Optimal Depth from Tuning
target_lambda = 0.05; 
lr_D = 0.0005;   % Slightly lower learning rate for smoother refinement
beta = 1e-4;     % Stability ridge
eps_val = 0.01;  
batchSize = 128; 
epochs = 500;     % Increased epochs for final convergence

% --- 3. DCT INITIALIZATION ---
D = zeros(m, n);
dct_basis = dctmtx(8);
count = 1;
for i = 1:8
    for j = 1:8
        atom = dct_basis(i,:)' * dct_basis(j,:);
        if count <= n
            D(:, count) = atom(:);
            count = count + 1;
        end
    end
end
D(:, 65:end) = randn(m, n-64) * 0.05;
D = D ./ sqrt(sum(D.^2, 1) + 1e-8);

% --- 4. TRAINING LOOP ---
mse_history = zeros(epochs, 1);
fprintf('Training Final Dictionary: K=%d, Lambda=%.2f\n', K, target_lambda);

for epoch = 1:epochs
    idx = randperm(N);
    epoch_loss = 0;
    
    for i = 1:batchSize:N-batchSize
        Y = Y_train(:, idx(i:i+batchSize-1));
        
        % FORWARD PASS
        x = zeros(n, batchSize);
        DtD = D' * D;
        DtY = D' * Y;
        I = eye(n);
        
        for k = 1:K
            % IRLS Weights for L1/2
            W_avg = mean((x.^2 + eps_val).^-0.75, 2);
            
            % Apply the fixed lambda for this layer
            L = DtD + target_lambda * diag(W_avg) + beta*I;
            x = L \ DtY;
        end
        
        % BACKWARD PASS (Dictionary Update)
        residual = (D * x) - Y;
        grad_D = (residual * x') / batchSize;
        
        % UPDATE
        D = D - lr_D * grad_D;
        D = D ./ sqrt(sum(D.^2, 1) + 1e-8); % Renormalize
        
        epoch_loss = epoch_loss + sum(residual.^2, 'all') / batchSize;
    end
    
    mse_history(epoch) = epoch_loss / (N/batchSize);
    if mod(epoch, 5) == 0 || epoch == 1
        fprintf('Epoch %d/%d | MSE: %.6f\n', epoch, epochs, mse_history(epoch));
    end
end

%% --- 5. RESULTS VISUALIZATION ---
figure('Color', 'w', 'Position', [100 100 1100 450]);

% Plot MSE
subplot(1,2,1);
plot(1:epochs, mse_history, 'Color', [0 0.447 0.741], 'LineWidth', 2);
grid on; box off;
xlabel('Training Epochs'); ylabel('Reconstruction MSE');
title(sprintf('Final Convergence (K=%d, \\lambda=%.2f)', K, target_lambda));

% Plot Dictionary
subplot(1,2,2);
display_dictionary_final(D);
title('Optimized L_{1/2} Dictionary Atoms');

% Save the model for testing
save('Trained_Unrolled_Net.mat', 'D', 'K', 'target_lambda');

%% --- HELPER ---
function display_dictionary_final(D)
    sz = 16; % 16x16 grid for 256 atoms
    big_img = zeros(sz*9, sz*9);
    for i = 1:256
        r = floor((i-1)/sz); c = mod(i-1, sz);
        atom = reshape(D(:,i), [8, 8]);
        atom = (atom - min(atom(:))) / (max(atom(:)) - min(atom(:)) + 1e-8);
        big_img(r*9+1:r*9+8, c*9+1:c*9+8) = atom;
    end
    imagesc(big_img); colormap gray; axis image off;
end