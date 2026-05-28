function [D, X] = ksvd_l12_train(Y, n_atoms, lambda, num_iters)
    % Y: Training patches (m x N) - typically 8x8 patches flattened to 64xN
    % n_atoms: Number of dictionary atoms to learn (e.g., 256)
    % lambda: Regularization parameter for IRLS
    % num_iters: Number of K-SVD iterations (typically 10-15)

    [m, N] = size(Y);
    
    % 1. Initialization: Pick random patches and normalize
    fprintf('Initializing Dictionary...\n');
    perm = randperm(N);
    D = Y(:, perm(1:n_atoms));
    D = D ./ sqrt(sum(D.^2)); % Column-wise normalization
    
    X = zeros(n_atoms, N);

    for iter = 1:num_iters
        fprintf('K-SVD Iteration %d/%d\n', iter, num_iters);
        
        % --- Step A: Sparse Coding (Parallelized for Speed) ---
        % We fix D and solve for X for all patches
        fprintf('  Sparse Coding Step...\n');
        parfor i = 1:N
            X(:, i) = irls_l12_stable(Y(:, i), D, lambda, 30, 1e-4);
        end
        
        % --- Step B: Dictionary Update Step ---
        fprintf('  Dictionary Update Step...\n');
        for k = 1:n_atoms
            % Identify indices of patches that use the k-th atom
            omega_k = find(X(k, :));
            
            if isempty(omega_k)
                % Dead Atom Handling: Replace with the worst-represented patch
                current_error = sum((Y - D*X).^2, 1);
                [~, max_err_idx] = max(current_error);
                new_atom = Y(:, max_err_idx);
                D(:, k) = new_atom / norm(new_atom);
                X(k, :) = 0; 
                continue;
            end
            
            % Compute the Restricted Error Matrix (Ek)
            % Only consider the columns (patches) that use atom d_k
            X_temp = X(:, omega_k);
            X_temp(k, :) = 0; % Remove current atom's contribution
            
            Ek_restricted = Y(:, omega_k) - D * X_temp;
            
            % Rank-1 Approximation via SVD
            [U, S, V] = svds(Ek_restricted, 1);
            
            % Update the atom and the non-zero coefficients
            D(:, k) = U;
            X(k, omega_k) = S * V';
        end
        
        % Monitor overall reconstruction error
        total_err = norm(Y - D*X, 'fro') / norm(Y, 'fro');
        fprintf('  Relative Frobenius Error: %.4f\n', total_err);
    end
end