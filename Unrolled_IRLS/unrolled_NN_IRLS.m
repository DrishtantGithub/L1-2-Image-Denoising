%% UNROLLED DICTIONARY LEARNING (L1/2-IRLS-Net)
clear; clc;

%% 1. PRE-PROCESS AND SAVE DATA PERMANENTLY
img_dir = 'datasets\BSDS500\images\train'; % Update this
files = dir(fullfile(img_dir, '*.jpg'));

patch_size = 8;
num_patches = 10000;
Y_train = zeros(patch_size^2, num_patches);

fprintf('Extracting patches...\n');
count = 1;
while count <= num_patches
    img = im2double(rgb2gray(imread(fullfile(files(randi(length(files))).folder, files(randi(length(files))).name))));
    [h, w] = size(img);
    r = randi(h-patch_size+1); c = randi(w-patch_size+1);
    p = img(r:r+patch_size-1, c:c+patch_size-1);
    p = p(:) - mean(p(:)); % DC removal
    if norm(p) > 1e-3
        Y_train(:, count) = p / norm(p);
        count = count + 1;
    end
end
save('BSDS500_Patches_Fixed.mat', 'Y_train');
fprintf('Data saved to BSDS500_Patches_Fixed.mat\n');
%% 2. UNROLLED L1/2-IRLS DICTIONARY LEARNING
clear; clc;

% Load Fixed Data
load('BSDS500_Patches_Fixed.mat'); % Y_train
[m, N] = size(Y_train);
numAtoms = 256;
K = 8; % Sweet spot for layers

% --- Parameters to Learn ---
% Initialize Dictionary with Unit Norm
D_init = randn(m, numAtoms);
D_init = D_init ./ sqrt(sum(D_init.^2, 1));

params.D = dlarray(D_init);
params.lambdas = dlarray(0.05 * ones(K, 1));
params.epsilons = dlarray(1e-4 * ones(K, 1));

% Optimization settings
lr = 0.001;
batchSize = 128;
numEpochs = 10;
m_state = []; v_state = []; % Adam states

fprintf('Starting training of Unrolled IRLS-Net...\n');

iteration = 0;
for epoch = 1:numEpochs
    idx = randperm(N);
    Y_train = Y_train(:, idx);
    
    for i = 1:batchSize:N-batchSize
        iteration = iteration + 1;
        Y_batch = dlarray(Y_train(:, i:i+batchSize-1));
        
        % Evaluate gradients
        [loss, grads] = dlfeval(@modelLoss, Y_batch, params, K);
        
        % Adam Update
        [params, m_state, v_state] = adamupdate(params, grads, m_state, v_state, iteration, lr);
        
        % CRITICAL: Normalize Dictionary after every update (KSVD constraint)
        params.D = params.D ./ sqrt(sum(params.D.^2, 1) + 1e-8);
        
        if mod(iteration, 100) == 0
            fprintf('Epoch %d | Iter %d | Loss: %.6f\n', epoch, iteration, double(loss));
        end
    end
end

%% --- LOSS FUNCTION ---
function [loss, gradients] = modelLoss(Y, params, K)
    % Extract raw data to strip labels but keep dlarray tracking
    Y_raw = extractdata(Y);
    D_raw = params.D;
    [m, batchSize] = size(Y_raw);
    dictSize = size(D_raw, 2);
    
    % Initial Sparse Code
    x = dlarray(zeros(dictSize, batchSize));
    
    % Identity for stability
    I = dlarray(eye(dictSize));
    
    % Unroll K IRLS layers
    for k = 1:K
        % 1. Weights from L1/2 quasi-norm
        W_vec = (x.^2 + params.epsilons(k)).^(-0.75);
        avg_W = mean(W_vec, 2); 
        
        % 2. Hessian: (D'D + lambda*W)
        DtD = D_raw' * D_raw;
        DtY = D_raw' * Y_raw;
        
        % Add weighted diagonal to DtD
        diag_idx = 1:(dictSize+1):numel(DtD);
        L = DtD;
        L(diag_idx) = L(diag_idx) + (params.lambdas(k) * avg_W(:)') + 1e-6;
        
        % 3. Differentiable Solve (mldivide on unformatted dlarray)
        x = L \ DtY;
    end
    
    % Reconstruction Error (Supervised)
    Y_hat = D_raw * x;
    loss = sum((Y_raw - Y_hat).^2, 'all') / batchSize;
    
    % Backprop
    gradients = dlgradient(loss, params);
end