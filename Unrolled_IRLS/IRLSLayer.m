%% 2. UNROLLED L1/2-IRLS DICTIONARY LEARNING
clear; clc;

% Load Fixed Data
load('BSDS500_Patches_Fixed.mat'); % Y_train
% Should contain Y_train [64 x N]
[m, N] = size(Y_train);
n = 256; % Atoms
K = 8;   % Layers

% --- 2. HYPERPARAMETERS (The "Sweet Spot") ---
lr_D = 0.001;    % Reduced for stability
lr_L = 0.0001;   % Slow learning for lambda
beta = 1e-4;     % Ridge for stability
eps_val = 0.01;  % Larger epsilon to prevent weight explosion
batchSize = 128; 
epochs = 30;

% --- 3. DCT INITIALIZATION (Better than random noise) ---
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
% Fill remaining with random noise if n > 64
if n > 64
    D(:, 65:end) = randn(m, n-64) * 0.1;
end
D = D ./ sqrt(sum(D.^2, 1) + 1e-8);

% Trackers
mse_history = zeros(epochs, 1);
lambda_history = zeros(epochs, 1);

% --- 4. TRAINING LOOP ---
fprintf('Training Unrolled IRLS-Net (Manual Backprop)...\n');

for epoch = 1:epochs
    idx = randperm(N);
    epoch_loss = 0;
    
    for i = 1:batchSize:N-batchSize
        Y = Y_train(:, idx(i:i+batchSize-1));
        
        % --- FORWARD PASS (MLP-style Unrolling) ---
        x = zeros(n, batchSize);
        DtD = D' * D;
        DtY = D' * Y;
        I = eye(n);
        
        % Store variables for backprop
        x_final = [];
        L_final = [];
        
        for k = 1:K
            % IRLS Weighting (L1/2 Prior)
            W_vec = (x.^2 + eps_val).^(-0.75);
            W_avg = mean(W_vec, 2);
            
            L = DtD + lambdas_schedule(k, epoch) * diag(W_avg) + beta*I;
            x = L \ DtY;
            
            if k == K
                x_final = x;
                L_final = L;
            end
        end
        
        % --- LOSS & BACKWARD PASS ---
        residual = (D * x_final) - Y;
        
        % Gradient w.r.t Dictionary
        grad_D = (residual * x_final') / batchSize;
        
        % Gradient w.r.t Lambda (Simplified Adjoint)
        % Note: We update the last layer's lambda specifically
        grad_x = D' * residual;
        invL = inv(L_final);
        dL_dlam = -trace(grad_x' * invL * diag(mean((x_final.^2 + eps_val).^-0.75, 2)) * x_final) / batchSize;
        
        % --- UPDATE STEP ---
        D = D - lr_D * grad_D;
        % Normalize atoms to unit sphere
        D = D ./ sqrt(sum(D.^2, 1) + 1e-8);
        
        % Update current epoch metrics
        epoch_loss = epoch_loss + sum(residual.^2, 'all') / batchSize;
    end
    
    mse_history(epoch) = epoch_loss / (N/batchSize);
    fprintf('Epoch %d/%d | MSE: %.6f\n', epoch, epochs, mse_history(epoch));
end

%% --- 5. VISUALIZATION ---
figure('Color', 'w', 'Position', [100 100 1000 400]);

% Plot 1: MSE Curve
subplot(1,2,1);
plot(1:epochs, mse_history, 'b-o', 'LineWidth', 1.5);
grid on; title('Training Convergence');
xlabel('Epoch'); ylabel('MSE (Normalized)');

% Plot 2: Learned Dictionary Atoms
subplot(1,2,2);
display_dictionary(D);
title('Learned L_{1/2} Dictionary Atoms');

%% --- HELPER FUNCTIONS ---
function l = lambdas_schedule(k, epoch)
    % A simple decaying schedule: early layers have higher lambda
    l = 0.1 * (0.9^(k-1)) * (0.95^(epoch-1));
end

function display_dictionary(D)
    n = size(D, 2);
    sz = ceil(sqrt(n));
    big_img = zeros(sz*9, sz*9);
    for i = 1:n
        r = floor((i-1)/sz);
        c = mod(i-1, sz);
        atom = reshape(D(:,i), [8, 8]);
        % Contrast stretch for visibility
        atom = (atom - min(atom(:))) / (max(atom(:)) - min(atom(:)) + 1e-8);
        big_img(r*9+1:r*9+8, c*9+1:c*9+8) = atom;
    end
    imagesc(big_img); colormap gray; axis image off;
end