function matRad_PGD_FMO_GPU()
%% matRad_PGD_FMO_GPU.m
% GPU-accelerated Projected Gradient Descent for Fluence Map Optimization
%
% This example demonstrates GPU acceleration for PGD optimization in
% radiation therapy treatment planning, comparing CPU vs GPU performance.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2017 the matRad development team. 
% 
% This file is part of the matRad project. It is subject to the license 
% terms in the LICENSE file found in the top-level directory of this 
% distribution and at https://github.com/e0404/matRad/LICENSE.md. No part 
% of the matRad project, including this file, may be copied, modified, 
% propagated, or distributed except according to the terms contained in the 
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matRad_cfg = matRad_rc;

% Test both CPU and GPU implementations
test_pgd_cpu_vs_gpu();

end

function test_pgd_cpu_vs_gpu()
%% Compare CPU vs GPU PGD optimization performance
    matRad_cfg = matRad_rc;
    
    % Problem setup - simulate fluence map optimization
    n_voxels = 10000;      % Reduced for better convergence
    n_beamlets = 2000;     % Reduced for better convergence
    n_iterations = 100;    % PGD iterations
    
    matRad_cfg.dispInfo('Setting up PGD optimization problem...\n');
    matRad_cfg.dispInfo('  Voxels: %d\n', n_voxels);
    matRad_cfg.dispInfo('  Beamlets: %d\n', n_beamlets);
    matRad_cfg.dispInfo('  Iterations: %d\n', n_iterations);
    
    % Generate synthetic dose influence matrix
    rng(42); % For reproducible results
    A = randn(n_voxels, n_beamlets, 'single'); % Use normal distribution
    A = abs(A); % Make positive (dose contributions are positive)
    A = A / 10; % Scale down for better conditioning
    
    % Create a more realistic target dose (some voxels need higher dose)
    d_target = ones(n_voxels, 1, 'single');
    % Make some target voxels require higher dose (tumor region)
    tumor_voxels = 1:round(n_voxels/4); % First 25% are "tumor"
    d_target(tumor_voxels) = 3.0; % Higher target dose for tumor
    
    % Weights (higher weight for tumor voxels)
    w = ones(n_voxels, 1, 'single');
    w(tumor_voxels) = 5.0; % Higher importance for tumor dose
    
    % PGD parameters - increased step size for better convergence
    step_size = 0.1; % Increased from 0.01
    max_fluence = 5.0; % Allow higher fluence
    
    % Initial fluence map (start from zero)
    x0 = zeros(n_beamlets, 1, 'single');
    
    matRad_cfg.dispInfo('\nRunning CPU-based PGD...\n');
    [x_cpu, obj_cpu, time_cpu] = pgd_cpu(A, d_target, w, x0, step_size, max_fluence, n_iterations);
    
    if gpuDeviceCount() > 0
        matRad_cfg.dispInfo('Running GPU-based PGD...\n');
        [x_gpu, obj_gpu, time_gpu] = pgd_gpu(A, d_target, w, x0, step_size, max_fluence, n_iterations);
        
        % Compare results
        matRad_cfg.dispInfo('\n=== Performance Comparison ===\n');
        matRad_cfg.dispInfo('CPU time: %.4f s\n', time_cpu);
        matRad_cfg.dispInfo('GPU time: %.4f s\n', time_gpu);
        matRad_cfg.dispInfo('Speedup: %.2fx\n', time_cpu / time_gpu);
        
        % Show convergence progress
        matRad_cfg.dispInfo('\n=== Convergence Analysis ===\n');
        matRad_cfg.dispInfo('Initial objective: %.6e\n', obj_cpu(1));
        matRad_cfg.dispInfo('Final CPU objective: %.6e\n', obj_cpu(end));
        matRad_cfg.dispInfo('Final GPU objective: %.6e\n', obj_gpu(end));
        matRad_cfg.dispInfo('CPU improvement: %.2f%%\n', 100*(obj_cpu(1)-obj_cpu(end))/obj_cpu(1));
        matRad_cfg.dispInfo('GPU improvement: %.2f%%\n', 100*(obj_gpu(1)-obj_gpu(end))/obj_gpu(1));
        
        % Verify solution quality
        diff_solution = norm(x_cpu - x_gpu) / norm(x_cpu);
        diff_objective = abs(obj_cpu(end) - obj_gpu(end)) / abs(obj_cpu(end));
        
        matRad_cfg.dispInfo('\n=== Solution Quality ===\n');
        matRad_cfg.dispInfo('Relative solution difference: %.6e\n', diff_solution);
        matRad_cfg.dispInfo('Relative objective difference: %.6e\n', diff_objective);
        
        % Check final dose delivery
        final_dose_cpu = A * x_cpu;
        final_dose_gpu = A * x_gpu;
        
        tumor_dose_error_cpu = mean(abs(final_dose_cpu(tumor_voxels) - d_target(tumor_voxels)));
        tumor_dose_error_gpu = mean(abs(final_dose_gpu(tumor_voxels) - d_target(tumor_voxels)));
        
        matRad_cfg.dispInfo('CPU tumor dose error: %.4f\n', tumor_dose_error_cpu);
        matRad_cfg.dispInfo('GPU tumor dose error: %.4f\n', tumor_dose_error_gpu);
        
        % Plot convergence curves
        plot_convergence(obj_cpu, obj_gpu, time_cpu, time_gpu);
        
    else
        matRad_cfg.dispInfo('GPU not available - CPU results only\n');
        matRad_cfg.dispInfo('CPU time: %.4f s\n', time_cpu);
        matRad_cfg.dispInfo('Final objective: %.6e\n', obj_cpu(end));
    end
end

function [x, objectives, total_time] = pgd_cpu(A, d_target, w, x0, step_size, max_fluence, n_iterations)
%% CPU implementation of Projected Gradient Descent
    
    x = x0;
    objectives = zeros(n_iterations, 1);
    
    tic;
    for iter = 1:n_iterations
        % Forward calculation: dose = A * x
        dose = A * x;
        
        % Residual: difference from target
        residual = dose - d_target;
        
        % Objective function: weighted least squares
        objectives(iter) = 0.5 * sum(w .* residual.^2);
        
        % Gradient calculation: grad = A' * (w .* residual)
        gradient = A' * (w .* residual);
        
        % Gradient step
        x = x - step_size * gradient;
        
        % Projection onto feasible set: x >= 0, x <= max_fluence
        x = max(0, min(max_fluence, x));
        
        % Progress display with more detail
        if mod(iter, 10) == 0
            improvement = 100*(objectives(1)-objectives(iter))/objectives(1);
            fprintf('  CPU Iteration %d: Objective = %.6e (%.2f%% improvement)\n', ...
                    iter, objectives(iter), improvement);
        end
    end
    total_time = toc;
end

function [x, objectives, total_time] = pgd_gpu(A, d_target, w, x0, step_size, max_fluence, n_iterations)
%% GPU implementation of Projected Gradient Descent
    
    % Transfer data to GPU
    A_gpu = gpuArray(A);
    d_target_gpu = gpuArray(d_target);
    w_gpu = gpuArray(w);
    x_gpu = gpuArray(x0);
    
    objectives = zeros(n_iterations, 1);
    
    tic;
    for iter = 1:n_iterations
        % Forward calculation: dose = A * x (on GPU)
        dose_gpu = A_gpu * x_gpu;
        
        % Residual: difference from target (on GPU)
        residual_gpu = dose_gpu - d_target_gpu;
        
        % Objective function: weighted least squares (on GPU)
        obj_gpu = 0.5 * sum(w_gpu .* residual_gpu.^2);
        objectives(iter) = gather(obj_gpu); % Transfer scalar back to CPU
        
        % Gradient calculation: grad = A' * (w .* residual) (on GPU)
        gradient_gpu = A_gpu' * (w_gpu .* residual_gpu);
        
        % Gradient step (on GPU)
        x_gpu = x_gpu - step_size * gradient_gpu;
        
        % Projection onto feasible set (on GPU)
        x_gpu = max(0, min(max_fluence, x_gpu));
        
        % Progress display with more detail
        if mod(iter, 10) == 0
            improvement = 100*(objectives(1)-objectives(iter))/objectives(1);
            fprintf('  GPU Iteration %d: Objective = %.6e (%.2f%% improvement)\n', ...
                    iter, objectives(iter), improvement);
        end
    end
    
    % Wait for GPU operations to complete
    wait(gpuDevice);
    total_time = toc;
    
    % Transfer final result back to CPU
    x = gather(x_gpu);
end

function plot_convergence(obj_cpu, obj_gpu, time_cpu, time_gpu)
%% Plot convergence comparison between CPU and GPU
    
    figure('Name', 'PGD Convergence: CPU vs GPU', 'Position', [100, 100, 1200, 400]);
    
    % Plot objective function convergence
    subplot(1, 2, 1);
    iter_cpu = 1:length(obj_cpu);
    iter_gpu = 1:length(obj_gpu);
    
    semilogy(iter_cpu, obj_cpu, 'b-', 'LineWidth', 2, 'DisplayName', 'CPU');
    hold on;
    semilogy(iter_gpu, obj_gpu, 'r--', 'LineWidth', 2, 'DisplayName', 'GPU');
    
    xlabel('Iteration');
    ylabel('Objective Function Value');
    title('Convergence Comparison');
    legend('Location', 'best');
    grid on;
    
    % Plot convergence vs wall-clock time
    subplot(1, 2, 2);
    time_per_iter_cpu = time_cpu / length(obj_cpu);
    time_per_iter_gpu = time_gpu / length(obj_gpu);
    
    time_axis_cpu = (0:length(obj_cpu)-1) * time_per_iter_cpu;
    time_axis_gpu = (0:length(obj_gpu)-1) * time_per_iter_gpu;
    
    semilogy(time_axis_cpu, obj_cpu, 'b-', 'LineWidth', 2, 'DisplayName', 'CPU');
    hold on;
    semilogy(time_axis_gpu, obj_gpu, 'r--', 'LineWidth', 2, 'DisplayName', 'GPU');
    
    xlabel('Wall-clock Time (s)');
    ylabel('Objective Function Value');
    title('Convergence vs Time');
    legend('Location', 'best');
    grid on;
    
    % Add speedup annotation
    speedup = time_cpu / time_gpu;
    annotation('textbox', [0.02, 0.85, 0.3, 0.1], ...
               'String', sprintf('GPU Speedup: %.2fx', speedup), ...
               'FontSize', 12, 'FontWeight', 'bold', ...
               'BackgroundColor', 'yellow', 'FitBoxToText', 'on');
end