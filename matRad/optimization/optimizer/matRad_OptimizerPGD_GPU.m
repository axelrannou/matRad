classdef matRad_OptimizerPGD_GPU < matRad_Optimizer
% matRad_OptimizerPGD_GPU: GPU-accelerated Projected Gradient Descent optimizer
%
% This class implements a GPU-accelerated Projected Gradient Descent (PGD) 
% algorithm for fluence map optimization in radiation therapy treatment planning.
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

    properties
        % Required abstract property from matRad_Optimizer
        options = struct(); % PGD-specific options structure
        
        % GPU-specific properties
        useGPU = true;
        gpuMemoryThreshold = 0.8; % Use max 80% of GPU memory
        
        % PGD algorithm parameters
        stepSize = 0.1;           % Initial step size
        adaptiveStepSize = true;  % Adaptive step size adjustment
        stepSizeReduction = 0.5;  % Factor for step size reduction
        minStepSize = 1e-10;       % Minimum step size
        maxFluence = 10.0;        % Maximum allowed fluence value
        
        % Convergence criteria
        relTolGrad = 1e-4;        % Relative gradient tolerance
        relTolObj = 1e-6;         % Relative objective tolerance
        maxIter = 1000;           % Maximum iterations
        
        % GPU arrays for persistent storage
        dij_gpu = [];
        w_gpu = [];
        d_ref_gpu = [];
        x_gpu = [];
        
        % GPU arrays for min/max constraints
        d_min_gpu = [];
        d_max_gpu = [];
        w_min_gpu = [];
        w_max_gpu = [];
        
        % GPU arrays for EUD objectives
        eud_a_gpu = [];
        eud_mask_gpu = [];
    end
    
    properties (SetAccess = protected)
        wResult     % Optimization result
        resultInfo  % Information about optimization result
    end
    
    methods
        function obj = matRad_OptimizerPGD_GPU()
            % Constructor
            obj@matRad_Optimizer();
            
            matRad_cfg = MatRad_Config.instance();
            
            obj.wResult = [];
            obj.resultInfo = [];
            
            % Check GPU availability
            if gpuDeviceCount() == 0
                matRad_cfg.dispWarning('No GPU devices found. Falling back to CPU mode.\n');
                obj.useGPU = false;
            else
                try
                    % Test GPU functionality
                    gpuDevice();
                    obj.useGPU = true;
                    matRad_cfg.dispInfo('GPU optimizer initialized successfully.\n');
                catch ME
                    matRad_cfg.dispWarning('GPU initialization failed: %s. Using CPU.\n', ME.message);
                    obj.useGPU = false;
                end
            end
        end
        
        function obj = optimize(obj, w0, optiProb, dij, cst)
            % Main optimization routine (required by matRad_Optimizer interface)
            
            matRad_cfg = MatRad_Config.instance();

            % User-configurable step size
            if isfield(optiProb, 'stepSize') && ~isempty(optiProb.stepSize)
                obj.stepSize = optiProb.stepSize;
            elseif isfield(optiProb, 'propOpt') && isfield(optiProb.propOpt, 'stepSize') && ~isempty(optiProb.propOpt.stepSize)
                obj.stepSize = optiProb.propOpt.stepSize;
            end
            
            % Initialize result info
            obj.resultInfo = struct();
            obj.resultInfo.iterations = 0;
            obj.resultInfo.status = 'running';
            
            if obj.useGPU
                obj = obj.optimizeGPU(w0, optiProb, dij, cst);
            else
                obj = obj.optimizeCPU(w0, optiProb, dij, cst);
            end
            
            % Cleanup GPU memory
            obj = obj.cleanupGPU();
        end
        
        function obj = optimizeGPU(obj, w0, optiProb, dij, cst)
            % GPU-accelerated optimization
            
            matRad_cfg = MatRad_Config.instance();
            matRad_cfg.dispInfo('Starting GPU-accelerated PGD optimization...\n');
            
            % Transfer data to GPU
            obj = obj.transferToGPU(w0, optiProb, dij, cst);
            
            % Setup objective parameters on GPU
            obj = obj.setupObjectiveParametersGPU(dij, cst);
            
            % Check if we have meaningful objectives
            max_ref_dose = gather(max(obj.d_ref_gpu));
            if max_ref_dose <= 0
                matRad_cfg.dispError('No target doses found! Cannot optimize without objectives.');
                obj.wResult = w0;
                obj.resultInfo.status = 'no_objectives';
                obj.resultInfo.iterations = 0;
                return;
            end
            
            % Initialize variables
            x = obj.x_gpu;
            step_size = gpuArray(single(obj.stepSize));
            
            % Pre-compute bounds on GPU
            lb_cpu = optiProb.lowerBounds(gather(x));
            ub_cpu = optiProb.upperBounds(gather(x));
            lb_gpu = gpuArray(single(lb_cpu));
            ub_gpu = gpuArray(single(ub_cpu));
            
            % Initialize objective history
            obj.resultInfo.objectiveHistory = zeros(obj.maxIter, 1);
            
            matRad_cfg.dispInfo('Starting GPU optimization loop...\n');
            
            % Main optimization loop - everything on GPU
            for iter = 1:obj.maxIter
                
                % Compute objective and gradient on GPU
                [obj_val_gpu, grad_gpu] = obj.computeObjectiveAndGradientGPU(x, optiProb, dij, cst);
                
                % Store iteration info (only transfer scalar)
                obj_val_cpu = gather(obj_val_gpu);
                if iter == 1
                    obj.resultInfo.initialObjective = obj_val_cpu;
                end
                obj.resultInfo.objectiveHistory(iter) = obj_val_cpu;
                
                % Check convergence
                if iter > 1
                    rel_obj_change = abs(obj.resultInfo.objectiveHistory(iter) - obj.resultInfo.objectiveHistory(iter-1)) / ...
                                   abs(obj.resultInfo.objectiveHistory(iter-1));
                    
                    if rel_obj_change < obj.relTolObj
                        matRad_cfg.dispInfo('Converged: relative objective change < %.2e\n', obj.relTolObj);
                        obj.resultInfo.status = 'converged';
                        break;
                    end
                end
                
                % Check for objective getting stuck at zero
                if iter > 10 && obj.resultInfo.objectiveHistory(iter) == 0
                    matRad_cfg.dispWarning('Objective stuck at zero - likely no meaningful targets\n');
                    obj.resultInfo.status = 'stuck_at_zero';
                    break;
                end
                
                % Adaptive step size (simple backtracking)
                if obj.adaptiveStepSize && iter > 1
                    if obj.resultInfo.objectiveHistory(iter) > obj.resultInfo.objectiveHistory(iter-1)
                        step_size = step_size * obj.stepSizeReduction;
                        if gather(step_size) < obj.minStepSize
                            matRad_cfg.dispInfo('Converged: step size too small\n');
                            obj.resultInfo.status = 'step_size_too_small';
                            break;
                        end
                    end
                end
                
                % Gradient step (on GPU)
                x = x - step_size .* grad_gpu;
                
                % Projection onto feasible set (on GPU)
                x = max(lb_gpu, min(ub_gpu, x));
                
                % Progress reporting (minimal CPU transfer)
                if mod(iter, 10) == 0 || iter == 1
                    improvement = 0;
                    if iter > 1 && obj.resultInfo.objectiveHistory(1) > 0
                        improvement = 100 * (obj.resultInfo.objectiveHistory(1) - obj.resultInfo.objectiveHistory(iter)) / obj.resultInfo.objectiveHistory(1);
                    end
                    grad_norm = gather(norm(grad_gpu));
                    matRad_cfg.dispInfo('Iter %3d: Obj = %.6e, Step = %.2e, GradNorm = %.2e, Improvement = %.2f%% [GPU]\n', ...
                        iter, obj.resultInfo.objectiveHistory(iter), gather(step_size), grad_norm, improvement);
                end
            end
            
            % Store final result (only one final transfer)
            obj.wResult = gather(x);
            obj.resultInfo.iterations = iter;
            obj.resultInfo.finalObjective = obj.resultInfo.objectiveHistory(iter);
            obj.resultInfo.objectiveHistory = obj.resultInfo.objectiveHistory(1:iter);
            
            if ~strcmp(obj.resultInfo.status, 'converged') && ~strcmp(obj.resultInfo.status, 'step_size_too_small') && ~strcmp(obj.resultInfo.status, 'stuck_at_zero')
                obj.resultInfo.status = 'max_iterations';
            end
            
            matRad_cfg.dispInfo('GPU optimization completed after %d iterations\n', iter);
        end
        
        function obj = optimizeCPU(obj, w0, optiProb, dij, cst)
            % Fallback CPU optimization
            
            matRad_cfg = MatRad_Config.instance();
            matRad_cfg.dispInfo('Starting CPU PGD optimization...\n');
            
            % Initialize
            x = w0;
            step_size = obj.stepSize;
            
            % Main optimization loop
            for iter = 1:obj.maxIter
                
                % Compute objective and gradient
                obj_val = optiProb.matRad_objectiveFunction(x, dij, cst);
                grad = optiProb.matRad_objectiveGradient(x, dij, cst);
                
                % Store iteration info
                if iter == 1
                    obj.resultInfo.initialObjective = obj_val;
                    obj.resultInfo.objectiveHistory = zeros(obj.maxIter, 1);
                end
                obj.resultInfo.objectiveHistory(iter) = obj_val;
                
                % Check convergence
                if iter > 1
                    rel_obj_change = abs(obj.resultInfo.objectiveHistory(iter) - obj.resultInfo.objectiveHistory(iter-1)) / ...
                                   abs(obj.resultInfo.objectiveHistory(iter-1));
                    
                    if rel_obj_change < obj.relTolObj
                        matRad_cfg.dispInfo('Converged: relative objective change < %.2e\n', obj.relTolObj);
                        obj.resultInfo.status = 'converged';
                        break;
                    end
                end
                
                % Adaptive step size
                if obj.adaptiveStepSize && iter > 1
                    if obj.resultInfo.objectiveHistory(iter) > obj.resultInfo.objectiveHistory(iter-1)
                        step_size = step_size * obj.stepSizeReduction;
                        if step_size < obj.minStepSize
                            matRad_cfg.dispInfo('Converged: step size too small\n');
                            obj.resultInfo.status = 'step_size_too_small';
                            break;
                        end
                    end
                end
                
                % Gradient step
                x = x - step_size * grad;
                
                % Projection onto feasible set
                lb = optiProb.lowerBounds(x);
                ub = optiProb.upperBounds(x);
                x = max(lb, min(ub, x));
                
                % Progress reporting
                if mod(iter, 10) == 0 || iter == 1
                    improvement = 0;
                    if iter > 1
                        improvement = 100 * (obj.resultInfo.objectiveHistory(1) - obj.resultInfo.objectiveHistory(iter)) / obj.resultInfo.objectiveHistory(1);
                    end
                    matRad_cfg.dispInfo('Iter %3d: Obj = %.6e, Step = %.2e, Improvement = %.2f%%\n', ...
                                      iter, obj.resultInfo.objectiveHistory(iter), step_size, improvement);
                end
            end
            
            % Store final result
            obj.wResult = x;
            obj.resultInfo.iterations = iter;
            obj.resultInfo.finalObjective = obj.resultInfo.objectiveHistory(iter);
            obj.resultInfo.objectiveHistory = obj.resultInfo.objectiveHistory(1:iter);
            
            if ~strcmp(obj.resultInfo.status, 'converged') && ~strcmp(obj.resultInfo.status, 'step_size_too_small')
                obj.resultInfo.status = 'max_iterations';
            end
            
            matRad_cfg.dispInfo('CPU optimization completed after %d iterations\n', iter);
        end
        
        function obj = transferToGPU(obj, w0, optiProb, dij, cst)
            % Transfer optimization data to GPU
            
            matRad_cfg = MatRad_Config.instance();
            matRad_cfg.dispInfo('Transferring data to GPU...\n');
            
            % Debug: Display dij structure
            matRad_cfg.dispInfo('dij structure fields: %s\n', strjoin(fieldnames(dij), ', '));
            
            % Handle dose influence matrix - could be cell array or matrix
            if isfield(dij, 'physicalDose')
                if iscell(dij.physicalDose)
                    % Handle cell array (multiple scenarios/fractions)
                    matRad_cfg.dispInfo('Detected cell array dij.physicalDose with %d elements\n', length(dij.physicalDose));
                    
                    % Check if it's a multi-scenario or single scenario in cell
                    if length(dij.physicalDose) == 1
                        % Single scenario wrapped in cell
                        if ~isempty(dij.physicalDose{1})
                            obj.dij_gpu = gpuArray(single(dij.physicalDose{1}));
                            matRad_cfg.dispInfo('Using single scenario from cell array\n');
                        else
                            matRad_cfg.dispError('Scenario in dij.physicalDose{1} is empty');
                        end
                    else
                        % Multiple scenarios - for now, use first non-empty one
                        scenario_found = false;
                        for i = 1:length(dij.physicalDose)
                            if ~isempty(dij.physicalDose{i})
                                obj.dij_gpu = gpuArray(single(dij.physicalDose{i}));
                                matRad_cfg.dispInfo('Using scenario %d out of %d for GPU optimization\n', i, length(dij.physicalDose));
                                scenario_found = true;
                                break;
                            end
                        end
                        if ~scenario_found
                            matRad_cfg.dispError('All scenarios in dij.physicalDose are empty');
                        end
                    end
                else
                    % Handle regular matrix
                    obj.dij_gpu = gpuArray(single(dij.physicalDose));
                    matRad_cfg.dispInfo('Using matrix dij.physicalDose directly\n');
                end
            else
                matRad_cfg.dispError('No physicalDose field found in dij structure');
            end
            
            % Transfer initial solution to GPU
            obj.x_gpu = gpuArray(single(w0));
            
            % Check GPU memory usage and display info
            gpu_info = gpuDevice();
            memory_used = (gpu_info.TotalMemory - gpu_info.AvailableMemory) / gpu_info.TotalMemory;
            
            matRad_cfg.dispInfo('GPU memory usage: %.1f%%\n', memory_used * 100);
            matRad_cfg.dispInfo('Dose matrix size on GPU: %dx%d\n', size(obj.dij_gpu, 1), size(obj.dij_gpu, 2));
            matRad_cfg.dispInfo('Initial solution size: %dx%d\n', size(obj.x_gpu, 1), size(obj.x_gpu, 2));
            
            if memory_used > obj.gpuMemoryThreshold
                matRad_cfg.dispWarning('High GPU memory usage detected. Consider reducing problem size.\n');
            end
        end
        
        function [obj_val, grad] = computeObjectiveAndGradientGPU(obj, x_gpu, optiProb, dij, cst)
            % Compute objective function and gradient directly on GPU
            % This handles both target doses and min/max constraints
            
            % Add MatRad_Config instance
            matRad_cfg = MatRad_Config.instance();
            
            % Compute dose on GPU: d = A * x
            dose_gpu = obj.dij_gpu * x_gpu;
            
            % Get reference dose and weights from CST (this could be optimized further)
            % For now, we'll get these once and cache them
            if isempty(obj.d_ref_gpu) || isempty(obj.w_gpu)
                obj = obj.setupObjectiveParametersGPU(dij, cst);
            end
            
            % Initialize objective and gradient
            obj_val = gpuArray(single(0));
            grad = gpuArray(zeros(size(x_gpu), 'single'));
            
            % 1. Main target objective: weighted least squares for target doses
            target_mask = obj.d_ref_gpu > 0;
            if any(target_mask)
                % Apply priority multiplier for target structures (PTV gets higher priority)
                target_weights = obj.w_gpu;
                target_weights(target_mask) = target_weights(target_mask) * 200; % 10x higher priority for targets
                
                residual_gpu = dose_gpu - obj.d_ref_gpu;
                weighted_residual_sq_gpu = target_weights .* (residual_gpu .^ 2);
                obj_val = obj_val + 0.5 * sum(weighted_residual_sq_gpu .* target_mask);
                
                % Gradient for target objective
                weighted_residual_gpu = target_weights .* residual_gpu .* target_mask;
                grad = grad + obj.dij_gpu' * weighted_residual_gpu;
            end
            
            % 2. Minimum dose constraints: penalize underdosage
            min_mask = obj.d_min_gpu > 0;
            if any(min_mask)
                underdose_gpu = max(gpuArray(single(0)), obj.d_min_gpu - dose_gpu);
                weighted_underdose_sq_gpu = obj.w_min_gpu .* (underdose_gpu .^ 2);
                obj_val = obj_val + 0.5 * sum(weighted_underdose_sq_gpu .* min_mask);
                
                % Gradient for minimum dose constraints (only where underdosed)
                underdose_mask = (dose_gpu < obj.d_min_gpu) & min_mask;
                if any(underdose_mask)
                    weighted_underdose_gpu = obj.w_min_gpu .* underdose_gpu .* underdose_mask;
                    grad = grad - obj.dij_gpu' * weighted_underdose_gpu; % Negative because we want to increase dose
                end
            end
            
            % 3. Maximum dose constraints: penalize overdosage
            max_mask = obj.d_max_gpu < inf;
            if any(max_mask)
                overdose_gpu = max(gpuArray(single(0)), dose_gpu - obj.d_max_gpu);
                weighted_overdose_sq_gpu = obj.w_max_gpu .* (overdose_gpu .^ 2);
                obj_val = obj_val + 0.5 * sum(weighted_overdose_sq_gpu .* max_mask);
                
                % Gradient for maximum dose constraints (only where overdosed)
                overdose_mask = (dose_gpu > obj.d_max_gpu) & max_mask;
                if any(overdose_mask)
                    weighted_overdose_gpu = obj.w_max_gpu .* overdose_gpu .* overdose_mask;
                    grad = grad + obj.dij_gpu' * weighted_overdose_gpu; % Positive because we want to decrease dose
                end
            end
            
            % 4. EUD objectives: Equivalent Uniform Dose (simplified as mean dose)
            if any(obj.eud_mask_gpu)
                % Group EUD objectives by structure (each structure should have same 'a' parameter for all its voxels)
                unique_structures = unique(obj.eud_a_gpu(obj.eud_mask_gpu));
                
                for a_val = unique_structures'
                    if a_val > 0
                        % Find voxels with this 'a' parameter (belonging to same structure)
                        current_eud_mask = obj.eud_mask_gpu & (obj.eud_a_gpu == a_val);
                        
                        if any(current_eud_mask)
                            % Get doses and target EUD for these voxels
                            doses_eud = dose_gpu(current_eud_mask);
                            target_eud = obj.d_ref_gpu(current_eud_mask);
                            weights_eud = obj.w_gpu(current_eud_mask);
                            
                            % Use first target EUD value (should be same for all voxels in structure)
                            target_eud_val = target_eud(1);
                            weight_val = weights_eud(1);
                            
                            % SIMPLIFIED: Treat EUD as mean dose for now
                            computed_mean_dose = mean(doses_eud);
                            
                            % Add EUD objective: 0.5 * weight * (mean_dose - target_EUD)^2
                            eud_diff = computed_mean_dose - target_eud_val;
                            obj_val = obj_val + 0.5 * weight_val * (eud_diff ^ 2);
                            
                            % Gradient: For mean dose, gradient is uniform across all voxels
                            % d(mean_dose)/d(dose_i) = 1/N for each voxel
                            N_voxels = length(doses_eud);
                            gradient_factor = weight_val * eud_diff / N_voxels;
                            
                            % Add EUD gradient contribution
                            eud_grad_full = gpuArray(zeros(size(dose_gpu), 'single'));
                            eud_grad_full(current_eud_mask) = gradient_factor;
                            
                            % Add to total gradient
                            grad = grad + obj.dij_gpu' * eud_grad_full;
                            
                        end
                    end
                end
            end
            
            % Add small regularization to prevent numerical issues
            reg_param = gpuArray(single(1e-8));
            obj_val = obj_val + 0.5 * reg_param * sum(x_gpu .^ 2);
            grad = grad + reg_param * x_gpu;
        end
        
        function obj = setupObjectiveParametersGPU(obj, dij, cst)
            % Setup reference dose and weights on GPU (called once)
            
            matRad_cfg = MatRad_Config.instance();
            matRad_cfg.dispInfo('Setting up objective parameters on GPU...\n');
            
            % Get matrix dimensions
            numVoxels = size(obj.dij_gpu, 1);
            
            % Initialize reference dose and weights on CPU first
            d_ref_cpu = zeros(numVoxels, 1, 'single');
            w_cpu = ones(numVoxels, 1, 'single');
            
            % For MinMaxDose, we need separate arrays for min and max constraints
            d_min_cpu = zeros(numVoxels, 1, 'single');  % Minimum dose constraints
            d_max_cpu = inf(numVoxels, 1, 'single');    % Maximum dose constraints
            w_min_cpu = zeros(numVoxels, 1, 'single');  % Weights for min constraints
            w_max_cpu = zeros(numVoxels, 1, 'single');  % Weights for max constraints
            
            % Initialize EUD parameters
            eud_a_cpu = zeros(numVoxels, 1, 'single');
            eud_mask_cpu = false(numVoxels, 1);
            
            % Extract from CST structure with proper matRad objective parsing
            for i = 1:size(cst, 1)
                if ~isempty(cst{i, 4}) && length(cst{i, 4}) >= 1 && ~isempty(cst{i, 4}{1})
                    indices = cst{i, 4}{1}; % Voxel indices for this structure
                    structName = cst{i, 2}; % Structure name
                    structType = cst{i, 3}; % Structure type (TARGET, OAR)
                    
                    % Process objectives for this structure
                    if size(cst, 2) >= 6 && ~isempty(cst{i, 6}) && length(cst{i, 6}) > 0
                        matRad_cfg.dispInfo('Processing structure "%s" (%s) with %d objectives\n', structName, structType, length(cst{i, 6}));
                        
                        for j = 1:length(cst{i, 6})
                            obj_func = cst{i, 6}{j};
                            
                            % Initialize values
                            target_dose = 0;
                            weight = 1;
                            
                            % Get penalty/weight
                            if isprop(obj_func, 'penalty') && ~isempty(obj_func.penalty)
                                weight = single(obj_func.penalty);
                            end
                            
                            % Extract parameters from matRad objective classes
                            if isprop(obj_func, 'parameters') && ~isempty(obj_func.parameters)
                                params = obj_func.parameters;
                                
                                % Debug: show what's in parameters
                                matRad_cfg.dispInfo('  Objective: %s\n', class(obj_func));
                                matRad_cfg.dispInfo('  Parameters cell array length: %d\n', length(params));
                                for k = 1:min(3, length(params))
                                    if isnumeric(params{k})
                                        matRad_cfg.dispInfo('    param{%d}: %.2f\n', k, params{k});
                                    else
                                        matRad_cfg.dispInfo('    param{%d}: %s\n', k, class(params{k}));
                                    end
                                end
                                
                                % Handle different objective types
                                if isa(obj_func, 'DoseConstraints.matRad_MinMaxDose')
                                    % MinMaxDose constraint: parameters are [minDose, maxDose, penalty_weight]
                                    if length(params) >= 2 && isnumeric(params{1}) && isnumeric(params{2})
                                        min_dose = single(params{1});
                                        max_dose = single(params{2});
                                        
                                        % Apply both minimum and maximum constraints
                                        d_min_cpu(indices) = min_dose;
                                        d_max_cpu(indices) = max_dose;
                                        w_min_cpu(indices) = weight;
                                        w_max_cpu(indices) = weight;
                                        
                                        matRad_cfg.dispInfo('  MinMaxDose: min=%.2f, max=%.2f Gy, weight=%.2e\n', min_dose, max_dose, weight);
                                        matRad_cfg.dispInfo('  Applied min constraint %.2f Gy and max constraint %.2f Gy to %d voxels\n', ...
                                                          min_dose, max_dose, length(indices));
                                        
                                        % For the main objective, use the target dose (middle of min/max range)
                                        target_dose = (min_dose + max_dose) / 2;
                                        d_ref_cpu(indices) = target_dose;
                                        w_cpu(indices) = weight;
                                        
                                        matRad_cfg.dispInfo('  Using target dose %.2f Gy (midpoint) for main objective\n', target_dose);
                                    end
                                    
                                elseif isa(obj_func, 'DoseObjectives.matRad_MaxDVH')
                                    % MaxDVH: typically [dose, volume_percent]
                                    if length(params) >= 1 && isnumeric(params{1})
                                        max_dose = single(params{1});
                                        volume_percent = 0;
                                        if length(params) >= 2 && isnumeric(params{2})
                                            volume_percent = params{2};
                                        end
                                        
                                        matRad_cfg.dispInfo('  MaxDVH: max dose %.2f Gy for %.1f%% volume\n', max_dose, volume_percent);
                                        
                                        % For MaxDVH, this is a MAXIMUM dose constraint, not a target
                                        d_max_cpu(indices) = max_dose;
                                        w_max_cpu(indices) = weight;
                                        
                                        matRad_cfg.dispInfo('  Applied max dose constraint %.2f Gy to %d voxels (no target dose)\n', max_dose, length(indices));
                                    end
                                    
                                elseif isa(obj_func, 'DoseObjectives.matRad_EUD')
                                    % EUD: typically [target_EUD, a_parameter]
                                    if length(params) >= 2 && isnumeric(params{1}) && isnumeric(params{2})
                                        target_eud = single(params{1});
                                        a_param = single(params{2});
                                        
                                        matRad_cfg.dispInfo('  EUD: target EUD %.2f Gy, a=%.2f\n', target_eud, a_param);
                                        
                                        % Store EUD parameters - we'll need special handling in objective computation
                                        d_ref_cpu(indices) = target_eud; % Store target EUD value
                                        w_cpu(indices) = weight;
                                        eud_a_cpu(indices) = a_param;
                                        eud_mask_cpu(indices) = true;
                                        
                                    elseif length(params) >= 1 && isnumeric(params{1})
                                        % Fallback if only target EUD is provided (assume a=1)
                                        target_eud = single(params{1});
                                        a_param = 1.0;
                                        matRad_cfg.dispInfo('  EUD: target EUD %.2f Gy, a=%.2f (default)\n', target_eud, a_param);
                                        
                                        d_ref_cpu(indices) = target_eud;
                                        w_cpu(indices) = weight;
                                        eud_a_cpu(indices) = a_param;
                                        eud_mask_cpu(indices) = true;
                                    end
                                    
                                else
                                    % Generic handling: try first parameter as dose
                                    if length(params) >= 1 && isnumeric(params{1})
                                        target_dose = single(params{1});
                                        matRad_cfg.dispInfo('  Generic objective: using first parameter %.2f as dose\n', target_dose);
                                        d_ref_cpu(indices) = target_dose;
                                        w_cpu(indices) = weight;
                                    end
                                end
                                
                                % Apply to main objective if we found a meaningful dose (and not already applied above)
                                if target_dose > 0 && ~isa(obj_func, 'DoseConstraints.matRad_MinMaxDose') && ...
                                   ~isa(obj_func, 'DoseObjectives.matRad_MaxDVH') && ~isa(obj_func, 'DoseObjectives.matRad_EUD')
                                    d_ref_cpu(indices) = target_dose;
                                    w_cpu(indices) = weight;
                                    matRad_cfg.dispInfo('  Applied dose=%.2f, weight=%.2e to %d voxels\n', ...
                                                      target_dose, weight, length(indices));
                                end
                            end
                        end
                    else
                        matRad_cfg.dispInfo('Structure "%s" has no objectives\n', structName);
                    end
                end
            end
            
            % Transfer to GPU - store all constraint types
            obj.d_ref_gpu = gpuArray(d_ref_cpu);
            obj.w_gpu = gpuArray(w_cpu);
            obj.d_min_gpu = gpuArray(d_min_cpu);
            obj.d_max_gpu = gpuArray(d_max_cpu);
            obj.w_min_gpu = gpuArray(w_min_cpu);
            obj.w_max_gpu = gpuArray(w_max_cpu);
            obj.eud_a_gpu = gpuArray(eud_a_cpu);
            obj.eud_mask_gpu = gpuArray(eud_mask_cpu);
            
            matRad_cfg.dispInfo('Final reference dose range: %.2f - %.2f\n', min(d_ref_cpu), max(d_ref_cpu));
            matRad_cfg.dispInfo('Final weight range: %.2e - %.2e\n', min(w_cpu), max(w_cpu));
            matRad_cfg.dispInfo('Voxels with target dose > 0: %d / %d\n', sum(d_ref_cpu > 0), length(d_ref_cpu));
            matRad_cfg.dispInfo('Voxels with min dose constraints: %d / %d\n', sum(d_min_cpu > 0), length(d_min_cpu));
            matRad_cfg.dispInfo('Voxels with max dose constraints: %d / %d\n', sum(d_max_cpu < inf), length(d_max_cpu));
            matRad_cfg.dispInfo('Voxels with EUD objectives: %d / %d\n', sum(eud_mask_cpu), length(eud_mask_cpu));
            
            % Show breakdown by structure type - only show structures that were actually processed
            target_voxels = 0;
            oar_voxels = 0;
            for i = 1:size(cst, 1)
                if ~isempty(cst{i, 4}) && length(cst{i, 4}) >= 1 && ~isempty(cst{i, 4}{1})
                    % Only show structures that actually had objectives in the CST
                    if size(cst, 2) >= 6 && ~isempty(cst{i, 6}) && length(cst{i, 6}) > 0
                        indices = cst{i, 4}{1};
                        if any(d_ref_cpu(indices) > 0)
                            if strcmp(cst{i, 3}, 'TARGET')
                                target_voxels = target_voxels + length(indices);
                                matRad_cfg.dispInfo('TARGET "%s": %d voxels with dose %.2f\n', cst{i, 2}, length(indices), d_ref_cpu(indices(1)));
                            else
                                oar_voxels = oar_voxels + length(indices);
                                matRad_cfg.dispInfo('OAR "%s": %d voxels with dose %.2f\n', cst{i, 2}, length(indices), d_ref_cpu(indices(1)));
                            end
                        end
                    end
                end
            end
            matRad_cfg.dispInfo('Summary: %d target voxels, %d OAR voxels with objectives\n', target_voxels, oar_voxels);
        end
        
        function obj = cleanupGPU(obj)
            % Clean up GPU memory
            
            if obj.useGPU
                obj.dij_gpu = [];
                obj.w_gpu = [];
                obj.d_ref_gpu = [];
                obj.x_gpu = [];
                obj.d_min_gpu = [];
                obj.d_max_gpu = [];
                obj.w_min_gpu = [];
                obj.w_max_gpu = [];
                obj.eud_a_gpu = [];
                obj.eud_mask_gpu = [];
                
                % Force garbage collection
                if gpuDeviceCount() > 0
                    try
                        reset(gpuDevice);
                    catch
                        % Ignore errors during cleanup
                    end
                end
            end
        end
        
        function [statusmsg, statusflag] = GetStatus(obj)
            % Get optimization status (required by matRad_Optimizer interface)
            
            try
                switch obj.resultInfo.status
                    case 'converged'
                        statusmsg = 'Optimization converged successfully';
                        statusflag = 1;
                    case 'max_iterations'
                        statusmsg = 'Maximum number of iterations reached';
                        statusflag = 0;
                    case 'step_size_too_small'
                        statusmsg = 'Step size became too small';
                        statusflag = 0;
                    case 'running'
                        statusmsg = 'Optimization is running';
                        statusflag = 0;
                    case 'no_objectives'
                        statusmsg = 'No optimization objectives found';
                        statusflag = 0;
                    case 'stuck_at_zero'
                        statusmsg = 'Objective function stuck at zero';
                        statusflag = 0;
                    otherwise
                        statusmsg = 'Unknown optimization status';
                        statusflag = -1;
                end
            catch
                statusmsg = 'No optimization status available';
                statusflag = -1;
            end
        end
    end
    
    methods (Static)
        function available = IsAvailable()
            % Check if GPU optimizer is available (required by matRad_Optimizer interface)
            available = gpuDeviceCount() > 0;
            
            if available
                try
                    % Test basic GPU functionality
                    gpuDevice();
                    test_array = gpuArray(rand(100));
                    result = test_array * test_array;
                    clear test_array result;
                catch
                    available = false;
                end
            end
        end
    end
end