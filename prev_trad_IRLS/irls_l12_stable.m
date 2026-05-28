function x = irls_l12_stable(y, D, lambda, max_iter, tol)
    % y: Observed patch (m x 1)
    % D: Dictionary (m x n)
    % lambda: Regularization parameter
    % max_iter: Maximum iterations (suggested 30-50)
    % tol: Convergence tolerance
    
    [m, n] = size(D);
    
    % Initialize x using back-projection
    x = D' * y; 
    
    % Stability epsilon (initial value)
    eps_val = 1.0; 
    
    % Precompute terms to speed up iterations
    DtD = D' * D;
    Dty = D' * y;
    
    for i = 1:max_iter
        x_prev = x;
        
        % 1. Weight Update (Stability Epsilon included)
        % For p = 1/2, w_i = (x_i^2 + eps)^((0.5-2)/2) = (x_i^2 + eps)^(-0.75)
        weights = (x.^2 + eps_val).^(-0.75);
        W = diag(weights);
        
        % Inside irls_l12_stable.m
        % 1. Increase the lower bound for epsilon
        eps_val = max(eps_val * 0.5, 1e-7); 
        
        % 2. Use a more robust solver than backslash (\) for ill-conditioned matrices
        % Replace: x = (DtD + lambda * W) \ Dty;
        % With a regularized version:
        L = DtD + lambda * W;
        x = (L + 1e-6 * eye(size(L))) \ Dty; % Adding a small ridge/Tikhonov term
        
        % 4. Convergence check
        if norm(x - x_prev) / (norm(x_prev) + 1e-12) < tol
            break;
        end
    end
end