%% Professional ML Visualization: Heatmap and Sensitivity Analysis
clear; clc; close all;

% Data Setup
K_vals = 2:10;
Lambda_vals = [0.01, 0.05, 0.10, 0.50, 1.00, 5.00];
MSE_results = [
    0.007353, 0.096827, 0.214915, 0.648258, 0.790700, 0.952881;
    0.006917, 0.097930, 0.219240, 0.647371, 0.789284, 0.950363;
    0.006858, 0.096392, 0.222478, 0.642019, 0.790658, 0.950929;
    0.006738, 0.098356, 0.211318, 0.650416, 0.796295, 0.952198;
    0.006790, 0.094156, 0.220546, 0.648851, 0.792607, 0.951788;
    0.006952, 0.093749, 0.223784, 0.648575, 0.790978, 0.953024;
    0.006804, 0.096712, 0.219412, 0.642782, 0.793848, 0.951271;
    0.006956, 0.092355, 0.222744, 0.649642, 0.794565, 0.951506;
    0.007018, 0.093285, 0.214650, 0.654686, 0.793406, 0.951428
];

figure('Color', 'w', 'Position', [100, 100, 1200, 500]);

% --- Plot 1: Hyperparameter Heatmap ---
% This is the industry standard for visualizing Grid Searches.
subplot(1, 2, 1);
h = heatmap(Lambda_vals, K_vals, MSE_results);
h.Title = 'Grid Search: Validation MSE Heatmap';
h.XLabel = 'Regularization (\lambda)';
h.YLabel = 'Network Depth (K)';
h.Colormap = sky; % 'Sky' or 'Parula' provides good contrast for MSE
h.ColorMethod = 'none'; % Show raw MSE values

% --- Plot 2: Sensitivity Analysis ---
% This shows how sensitive the model is to depth (K) for each lambda setting.
subplot(1, 2, 2);
hold on;
colors = lines(length(Lambda_vals));
for j = 1:length(Lambda_vals)
    plot(K_vals, MSE_results(:, j), '-o', 'Color', colors(j,:), ...
        'LineWidth', 1.5, 'MarkerFaceColor', colors(j,:));
end
grid on;
set(gca, 'YScale', 'log'); % Crucial for seeing differences in small MSE values
xlabel('Network Depth (K)');
ylabel('Validation MSE (Log Scale)');
title('Parameter Sensitivity Analysis');
legend(string(Lambda_vals), 'Location', 'northeastoutside');
title(legend, '\lambda');