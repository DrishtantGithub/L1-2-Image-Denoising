%% Phase 1 Sanity Check
m = 64; n = 128; % Dictionary dimensions
k = 5;           % True sparsity level

% 1. Generate Overcomplete DCT Dictionary
D = dctmtx(n); D = D(1:m, :); 
D = D ./ sqrt(sum(D.^2)); % Normalize columns

% 2. Generate Sparse Signal
x_true = zeros(n, 1);
p = randperm(n);
x_true(p(1:k)) = randn(k, 1);
y = D * x_true + 0.01 * randn(m, 1); % Add noise

% 3. Run IRLS Solver
lambda = 0.1;
x_rec = irls_l12_stable(y, D, lambda, 50, 1e-6);

% 4. Plot Comparison
figure;
subplot(2,1,1); stem(x_true); title('Ground Truth (Sparse)');
subplot(2,1,2); stem(x_rec); title('L_{1/2} Recovery via IRLS');