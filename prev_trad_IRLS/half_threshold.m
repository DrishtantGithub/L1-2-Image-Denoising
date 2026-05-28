function x_hat = half_threshold(b, lambda)
    % Analytical solution for L1/2 proximal mapping
    threshold = (54^(1/3) / 4) * (lambda^(2/3));
    x_hat = zeros(size(b));
    
    idx = abs(b) > threshold;
    if any(idx)
        b_sub = b(idx);
        % phi term from Equation 20 in Xu et al.
        phi = acos((lambda/8) * (abs(b_sub)/3).^(-1.5));
        
        % Half-thresholding formula
        x_hat(idx) = (2/3) * b_sub .* (1 + cos((2*pi/3) - (2/3)*phi));
    end
end