classdef matRad_OptimizerEclipseBased < matRad_Optimizer
    % matRad_OptimizerEclipseBased
    % Eclipse-based optimizer using L-BFGS-B (Limited-memory BFGS with Bounds)
    % Similar to Varian Eclipse's optimization algorithm
    %
    % References:
    %   [1] Nocedal & Wright, Numerical Optimization (2006)
    %   [2] Byrd et al., SIAM J. Sci. Comput. 16(5):1190-1208 (1995)
    %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Copyright 2024 the matRad development team.
    %
    % This file is part of the matRad project. It is subject to the license
    % terms in the LICENSE file found in the top-level directory of this
    % distribution and at https://github.com/e0404/matRad/LICENSE.md.
    %
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    properties
        options = struct();
        
        % L-BFGS parameters
        maxMemory = 10;           % Number of correction pairs to store (L-BFGS memory)
        lineSearchMaxIter = 20;   % Maximum line search iterations
        c1 = 1e-4;                % Armijo condition parameter
        c2 = 0.9;                 % Wolfe condition parameter (curvature)
        initialStepSize = 1.0;    % Initial step size for line search
        useStrongWolfe = true;    % Use strong Wolfe conditions
        minGradNorm = 1e-5;       % Minimum projected gradient norm
        adaptiveRestart = true;   % Adaptive restart based on progress
        gradientCheckFreq = 10;   % Check gradient quality every N iterations
        
        % Convergence
        relTolGrad = 1e-5;        % Tighter gradient tolerance
        relTolObj = 1e-7;         % Tighter objective tolerance for better convergence
        maxIter = 5000;            % Increased from 200 for better results
        maxFluence = 10.0;
        
        % Hotspot control (Eclipse-style NTO)
        hotspotPenalty = 10000;      % Penalty weight for hotspots
        hotspotThreshold = 1.00;   % 102% of prescription dose triggers hotspot penalty
        prescriptionDose = 0;      % Will be set from target objectives
        
        % CPU storage (from PGD)
        dij_cpu = [];
        d_ref_cpu = [];
        w_cpu = [];
        x_cpu = [];
        
        % Min/Max constraints
        d_min_cpu = [];
        d_max_cpu = [];
        w_min_cpu = [];
        w_max_cpu = [];
        
        % EUD
        eud_a_cpu = [];
        eud_mask_cpu = [];
        
        % Precomputed masks
        target_mask = [];
        min_mask = [];
        max_mask = [];
        
        useSinglePrecision = false;
        progressEvery = 10;
    end
    
    properties (SetAccess = protected)
        sHistory = {};            % History of s vectors (x_k+1 - x_k)
        yHistory = {};            % History of y vectors (grad_k+1 - grad_k)
        rhoHistory = [];          % History of 1/(y'*s) values
        currentMemorySize = 0;    % Current number of stored correction pairs
        wResult
        resultInfo
    end
    
    methods
        function obj = matRad_OptimizerEclipseBased()
            obj@matRad_Optimizer();
            obj.wResult = [];
            obj.resultInfo = [];
        end
        
        function obj = optimize(obj, w0, optiProb, dij, cst)
            % Main optimization loop
            % Input:
            %   w0: Initial parameter vector (beamlet weights)
            %   optiProb: Optimization problem structure
            %   dij: Dose influence matrix
            %   cst: Cell structure table
            % Output:
            %   obj: Updated optimizer object with wResult and resultInfo
            
            matRad_cfg = MatRad_Config.instance();
            
            obj.resultInfo = struct();
            obj.resultInfo.iterations = 0;
            obj.resultInfo.status = 'running';
            
            % Start timing
            optimization_start_time = tic;
            
            % Prepare CPU data and objective params
            obj = obj.transferToCPU(w0, optiProb, dij, cst);
            obj = obj.setupObjectiveParametersCPU(dij, cst);
            
            % Basic validation
            max_ref_dose = max(obj.d_ref_cpu(:));
            if isempty(max_ref_dose) || max_ref_dose <= 0
                matRad_cfg.dispError('No target doses found! Cannot optimize without objectives.');
                obj.wResult = w0;
                obj.resultInfo.status = 'no_objectives';
                obj.resultInfo.iterations = 0;
                obj.resultInfo.optimizationTime = toc(optimization_start_time);
                return;
            end
            
            % Initialize L-BFGS
            obj.sHistory = {};
            obj.yHistory = {};
            obj.rhoHistory = [];
            obj.currentMemorySize = 0;
            
            % Initialize
            x = obj.x_cpu;
            n = length(x);
            
            % Get bounds
            lb = zeros(n, 1);  % Lower bound (non-negative fluences)
            ub = obj.maxFluence * ones(n, 1);  % Upper bound
            
            % Project initial point onto feasible region
            x = max(lb, min(ub, x));
            
            % Evaluate initial objective and gradient
            [fval, grad] = obj.computeObjectiveAndGradientCPU(x);
            
            % Initialize history
            obj.resultInfo.objectiveHistory = zeros(obj.maxIter, 1);
            obj.resultInfo.objectiveHistory(1) = fval;
            obj.resultInfo.initialObjective = fval;
            
            % Display header
            matRad_cfg.dispInfo('\nEclipse-based L-BFGS-B Optimizer (Memory: %d)\n', obj.maxMemory);
            matRad_cfg.dispInfo('%5s %12s %12s %12s %10s\n', ...
                'Iter', 'Objective', 'ProjGrad', 'Step Size', 'LS Iters');
            matRad_cfg.dispInfo('%s\n', repmat('-', 1, 65));
            
            % Main optimization loop
            for iter = 1:obj.maxIter
                obj.resultInfo.iterations = iter;
                
                % Compute projected gradient norm for convergence check
                projGradNorm = obj.computeProjectedGradNorm(x, grad, lb, ub);
                
                % Check convergence
                if projGradNorm < obj.relTolGrad || projGradNorm < obj.minGradNorm
                    obj.resultInfo.status = 'converged';
                    matRad_cfg.dispInfo('%5d %12.5e %12.5e (Converged)\n', ...
                        iter, fval, projGradNorm);
                    break;
                end
                
                % Check relative objective change
                if iter > 1
                    prev = obj.resultInfo.objectiveHistory(iter-1);
                    if prev ~= 0
                        rel_obj_change = abs(fval - prev) / abs(prev);
                        if rel_obj_change < obj.relTolObj
                            matRad_cfg.dispInfo('%5d %12.5e %12.5e (RelObj < %.2e)\n', ...
                                iter, fval, projGradNorm, obj.relTolObj);
                            obj.resultInfo.status = 'converged';
                            break;
                        end
                    end
                end
                
                % Compute search direction using L-BFGS two-loop recursion
                searchDir = obj.computeLBFGSDirection(grad);
                
                % Project search direction (Cauchy point approach for bound constraints)
                searchDir = obj.projectSearchDirection(x, searchDir, grad, lb, ub);
                
                % Line search with Wolfe conditions
                [alpha, fval_new, grad_new, lsIter] = obj.lineSearch(...
                    x, fval, grad, searchDir, lb, ub);
                
                % Display iteration info
                if mod(iter, obj.progressEvery) == 0 || iter == 1
                    improvement = 0;
                    if obj.resultInfo.objectiveHistory(1) > 0
                        improvement = 100 * (obj.resultInfo.objectiveHistory(1) - fval) / obj.resultInfo.objectiveHistory(1);
                    end
                    matRad_cfg.dispInfo('%5d %12.5e %12.5e %12.5e %10d (Imp: %.2f%%)\n', ...
                        iter, fval, projGradNorm, alpha, lsIter, improvement);
                end
                
                % Update position
                x_new = max(lb, min(ub, x + alpha * searchDir));
                
                % Update L-BFGS history
                s = x_new - x;
                y = grad_new - grad;
                
                % Check curvature condition before updating
                sDotY = s' * y;
                if sDotY > 1e-10 * norm(s) * norm(y)
                    obj.updateLBFGSHistory(s, y);
                end
                
                % Update current point
                x = x_new;
                fval = fval_new;
                grad = grad_new;
                
                % Store objective
                obj.resultInfo.objectiveHistory(iter+1) = fval;
                
                % Adaptive restart if no progress
                if obj.adaptiveRestart && iter > 10
                    recentProgress = abs(obj.resultInfo.objectiveHistory(max(1,iter-5)) - fval) / ...
                        (abs(obj.resultInfo.objectiveHistory(max(1,iter-5))) + 1e-10);
                    if recentProgress < 1e-6
                        obj.resetLBFGSHistory();
                        matRad_cfg.dispInfo('      (L-BFGS history reset)\n');
                    end
                end
            end
            
            % Finalize
            obj.resultInfo.optimizationTime = toc(optimization_start_time);
            obj.wResult = x;
            obj.resultInfo.finalObjective = fval;
            obj.resultInfo.objectiveHistory = obj.resultInfo.objectiveHistory(1:obj.resultInfo.iterations);
            
            if ~strcmp(obj.resultInfo.status, 'converged')
                obj.resultInfo.status = 'max_iterations';
            end
            
            % Format time display
            total_seconds = obj.resultInfo.optimizationTime;
            hours = floor(total_seconds / 3600);
            minutes = floor(mod(total_seconds, 3600) / 60);
            seconds = mod(total_seconds, 60);
            
            matRad_cfg.dispInfo('\nOptimization completed after %d iterations in %dh %dm %.2fs\n', ...
                obj.resultInfo.iterations, hours, minutes, seconds);
            matRad_cfg.dispInfo('Final objective: %.6e\n', fval);
        end
        
        function dir = computeLBFGSDirection(obj, grad)
            % Two-loop recursion for L-BFGS direction computation
            
            q = grad;
            alphas = zeros(obj.currentMemorySize, 1);
            
            % First loop (backward)
            for i = obj.currentMemorySize:-1:1
                alphas(i) = obj.rhoHistory(i) * (obj.sHistory{i}' * q);
                q = q - alphas(i) * obj.yHistory{i};
            end
            
            % Initial Hessian approximation (scaling)
            if obj.currentMemorySize > 0
                % Use Barzilai-Borwein-like scaling
                gamma = (obj.sHistory{end}' * obj.yHistory{end}) / ...
                    (obj.yHistory{end}' * obj.yHistory{end});
                r = gamma * q;
            else
                % Identity matrix (steepest descent for first iteration)
                r = q;
            end
            
            % Second loop (forward)
            for i = 1:obj.currentMemorySize
                beta = obj.rhoHistory(i) * (obj.yHistory{i}' * r);
                r = r + obj.sHistory{i} * (alphas(i) - beta);
            end
            
            % Search direction is negative of the result
            dir = -r;
        end
        
        function dir = projectSearchDirection(obj, w, dir, grad, lb, ub)
            % Project search direction for bound-constrained optimization
            % Uses generalized Cauchy point approach
            
            % Find breakpoints where variables hit bounds
            t = inf(length(w), 1);
            
            for i = 1:length(w)
                if dir(i) < 0
                    t(i) = (lb(i) - w(i)) / dir(i);
                elseif dir(i) > 0
                    t(i) = (ub(i) - w(i)) / dir(i);
                end
            end
            
            % Sort breakpoints
            [t_sorted, idx] = sort(t);
            
            % Initialize projected direction
            dirProj = dir;
            
            % Set direction to zero for variables at bounds moving in infeasible direction
            for i = 1:length(w)
                if w(i) <= lb(i) + eps && dir(i) < 0
                    dirProj(i) = 0;
                elseif w(i) >= ub(i) - eps && dir(i) > 0
                    dirProj(i) = 0;
                end
            end
            
            dir = dirProj;
        end
        
        function [alpha, fval_new, grad_new, iter] = lineSearch(obj, w, fval, grad, dir, lb, ub)
            % Line search with Wolfe conditions and bound constraints
            
            alpha = obj.initialStepSize;
            alpha_low = 0;
            alpha_high = inf;
            
            % Directional derivative
            dirDeriv = grad' * dir;
            
            % If direction is not a descent direction, return step size 0
            if dirDeriv >= 0
                alpha = 0;
                fval_new = fval;
                grad_new = grad;
                iter = 0;
                return;
            end
            
            for iter = 1:obj.lineSearchMaxIter
                % Compute trial point
                w_new = max(lb, min(ub, w + alpha * dir));
                
                % Evaluate objective and gradient
                [fval_new, grad_new] = obj.computeObjectiveAndGradientCPU(w_new);
                
                % Check Armijo condition (sufficient decrease)
                if fval_new > fval + obj.c1 * alpha * dirDeriv
                    % Armijo condition violated, reduce step size
                    alpha_high = alpha;
                    alpha = 0.5 * (alpha_low + alpha_high);
                    continue;
                end
                
                % Check Wolfe curvature condition
                dirDeriv_new = grad_new' * dir;
                
                if obj.useStrongWolfe
                    % Strong Wolfe condition
                    if abs(dirDeriv_new) <= -obj.c2 * dirDeriv
                        break;  % Both conditions satisfied
                    end
                else
                    % Standard Wolfe condition
                    if dirDeriv_new >= obj.c2 * dirDeriv
                        break;  % Both conditions satisfied
                    end
                end
                
                % Curvature condition not satisfied
                if dirDeriv_new < 0
                    % Can increase step size
                    alpha_low = alpha;
                    if isinf(alpha_high)
                        alpha = 2 * alpha;
                    else
                        alpha = 0.5 * (alpha_low + alpha_high);
                    end
                else
                    % Need to decrease step size
                    alpha_high = alpha;
                    alpha = 0.5 * (alpha_low + alpha_high);
                end
            end
        end
        
        function updateLBFGSHistory(obj, s, y)
            % Update L-BFGS history with new correction pair
            
            rho = 1.0 / (y' * s);
            
            if obj.currentMemorySize < obj.maxMemory
                % Add new pair
                obj.currentMemorySize = obj.currentMemorySize + 1;
                obj.sHistory{obj.currentMemorySize} = s;
                obj.yHistory{obj.currentMemorySize} = y;
                obj.rhoHistory(obj.currentMemorySize) = rho;
            else
                % Remove oldest pair and add new one
                obj.sHistory = [obj.sHistory(2:end), {s}];
                obj.yHistory = [obj.yHistory(2:end), {y}];
                obj.rhoHistory = [obj.rhoHistory(2:end), rho];
            end
        end
        
        function resetLBFGSHistory(obj)
            % Reset L-BFGS history (adaptive restart)
            obj.sHistory = {};
            obj.yHistory = {};
            obj.rhoHistory = [];
            obj.currentMemorySize = 0;
        end
        
        function projGradNorm = computeProjectedGradNorm(obj, w, grad, lb, ub)
            % Compute norm of projected gradient (for convergence check)
            
            % Projected gradient: g_P(x) = x - P(x - grad)
            projGrad = w - max(lb, min(ub, w - grad));
            projGradNorm = norm(projGrad, inf);
        end
        
        function obj = transferToCPU(obj, w0, ~, dij, ~)
            matRad_cfg = MatRad_Config.instance();
            matRad_cfg.dispInfo('Preparing data for CPU...\n');

            % Dose influence matrix: keep sparse if already sparse
            if isfield(dij, 'physicalDose')
                if iscell(dij.physicalDose)
                    scenario_found = false;
                    for i = 1:numel(dij.physicalDose)
                        if ~isempty(dij.physicalDose{i})
                            M = dij.physicalDose{i};
                            scenario_found = true;
                            break;
                        end
                    end
                    if ~scenario_found
                        matRad_cfg.dispError('All scenarios in dij.physicalDose are empty');
                    end
                else
                    M = dij.physicalDose;
                end
            else
                matRad_cfg.dispError('No physicalDose field found in dij structure');
            end

            % Cast types without destroying sparsity
            if obj.useSinglePrecision
                if issparse(M), obj.dij_cpu = sparse(single(M)); else, obj.dij_cpu = single(M); end
                obj.x_cpu = single(w0);
            else
                if issparse(M), obj.dij_cpu = sparse(double(M)); else, obj.dij_cpu = double(M); end
                obj.x_cpu = double(w0);
            end

            matRad_cfg.dispInfo('Dose matrix size: %dx%d (sparse=%d)\n', size(obj.dij_cpu,1), size(obj.dij_cpu,2), issparse(obj.dij_cpu));
            matRad_cfg.dispInfo('Initial solution size: %dx%d\n', size(obj.x_cpu,1), size(obj.x_cpu,2));
        end

        function [obj_val, grad] = computeObjectiveAndGradientCPU(obj, x)
            % Compute dose (1 SpMV)
            dose = obj.dij_cpu * x;

            if isempty(obj.d_ref_cpu) || isempty(obj.w_cpu)
                error('Objective parameters not initialized');
            end

            % Accumulate voxel-space gradient contributions, then do 1 A' multiply.
            nVox = size(obj.dij_cpu,1);
            g_vox = zeros(nVox,1, class(x));
            obj_val = 0;

            % 1) Target objective (weighted least squares)
            if any(obj.target_mask)
                idx = obj.target_mask;
                target_weights = obj.w_cpu;
                target_weights(idx) = target_weights(idx) * 200; % priority
                residual = dose(idx) - obj.d_ref_cpu(idx);

                obj_val = obj_val + 0.5 * sum(target_weights(idx) .* (residual .^ 2));
                g_vox(idx) = g_vox(idx) + target_weights(idx) .* residual;
            end

            % 2) Minimum dose constraints with Eclipse-style adaptive penalty
            if any(obj.min_mask)
                idx = obj.min_mask;
                underdose = obj.d_min_cpu(idx) - dose(idx);
                active = underdose > 0;
                if any(active)
                    u_idx = find(idx); u_idx = u_idx(active);
                    u_val = underdose(active);
                    wv = obj.w_min_cpu(u_idx);
                    
                    % Eclipse-style adaptive penalty: scale by violation severity
                    % Larger violations get exponentially higher penalties
                    violation_ratio = u_val ./ obj.d_min_cpu(u_idx);
                    adaptive_scale = 1 + 100 * (violation_ratio .^ 2);
                    wv = wv .* adaptive_scale;
                    
                    obj_val = obj_val + 0.5 * sum(wv .* (u_val .^ 2));
                    g_vox(u_idx) = g_vox(u_idx) - wv .* u_val;
                end
            end

            % 3) Maximum dose constraints
            if any(obj.max_mask)
                idx = obj.max_mask;
                overdose = dose(idx) - obj.d_max_cpu(idx);
                active = overdose > 0;
                if any(active)
                    o_idx = find(idx); o_idx = o_idx(active);
                    o_val = overdose(active);
                    wv = obj.w_max_cpu(o_idx);
                    obj_val = obj_val + 0.5 * sum(wv .* (o_val .^ 2));
                    g_vox(o_idx) = g_vox(o_idx) + wv .* o_val;
                end
            end

            % 5) Eclipse-style Global Hotspot Penalty (NTO-like)
            % Aggressively penalize doses exceeding prescription * threshold
            if obj.prescriptionDose > 0 && obj.hotspotPenalty > 0
                hotspotLimit = obj.prescriptionDose * obj.hotspotThreshold;
                hotspotExcess = dose - hotspotLimit;
                hotspotActive = hotspotExcess > 0;
                
                if any(hotspotActive)
                    % Progressive penalty: small violations get moderate penalty,
                    % large violations get much stronger penalty (cubic scaling)
                    excess_vals = hotspotExcess(hotspotActive);
                    normalized_excess = excess_vals / obj.prescriptionDose;
                    
                    % Cubic penalty for stronger hotspot suppression
                    penalty_vals = obj.hotspotPenalty * (normalized_excess .^ 2) .* (1 + 10 * normalized_excess);
                    
                    obj_val = obj_val + sum(penalty_vals .* (excess_vals .^ 2));
                    
                    % Gradient: d/d(dose) of penalty * excess^2
                    % = 2 * penalty * excess + penalty' * excess^2
                    % Simplified: use quadratic gradient with scaling
                    grad_scale = 2 * obj.hotspotPenalty * normalized_excess .* (1 + 15 * normalized_excess);
                    g_vox(hotspotActive) = g_vox(hotspotActive) + grad_scale .* excess_vals;
                end
            end

            % 4) EUD objectives (simplified as mean dose per group)
            if any(obj.eud_mask_cpu)
                a_vals = unique(obj.eud_a_cpu(obj.eud_mask_cpu));
                for a_val = a_vals.'
                    if a_val <= 0, continue; end
                    grp = obj.eud_mask_cpu & (obj.eud_a_cpu == a_val);
                    if ~any(grp), continue; end
                    doses_eud = dose(grp);
                    target_eud = obj.d_ref_cpu(grp);
                    weights_eud = obj.w_cpu(grp);

                    % assume single target/weight per group
                    target_eud_val = target_eud(1);
                    weight_val = weights_eud(1);

                    mean_dose = mean(doses_eud);
                    eud_diff = mean_dose - target_eud_val;

                    obj_val = obj_val + 0.5 * weight_val * (eud_diff ^ 2);

                    N = nnz(grp);
                    gradient_factor = weight_val * eud_diff / N;
                    g_vox(grp) = g_vox(grp) + gradient_factor;
                end
            end

            % One A' multiply for gradient (1 SpMV^T)
            grad = obj.dij_cpu' * g_vox;

            % Small L2 regularization
            reg_param = cast(1e-8, class(x));
            obj_val = obj_val + 0.5 * reg_param * sum(x .^ 2);
            grad = grad + reg_param * x;
        end

        function obj = setupObjectiveParametersCPU(obj, ~, cst)
            matRad_cfg = MatRad_Config.instance();
            matRad_cfg.dispInfo('Setting up objective parameters on CPU...\n');

            numVoxels = size(obj.dij_cpu, 1);

            d_ref = zeros(numVoxels, 1);
            w = ones(numVoxels, 1);

            d_min = zeros(numVoxels, 1);
            d_max = inf(numVoxels, 1);
            w_min = zeros(numVoxels, 1);
            w_max = zeros(numVoxels, 1);

            eud_a = zeros(numVoxels, 1);
            eud_mask = false(numVoxels, 1);

            for i = 1:size(cst, 1)
                if ~isempty(cst{i,4}) && numel(cst{i,4}) >= 1 && ~isempty(cst{i,4}{1})
                    indices = cst{i,4}{1};
                    structName = cst{i,2};
                    structType = cst{i,3};

                    if size(cst,2) >= 6 && ~isempty(cst{i,6}) && numel(cst{i,6}) > 0
                        matRad_cfg.dispInfo('Processing structure "%s" (%s) with %d objectives\n', structName, structType, numel(cst{i,6}));

                        for j = 1:numel(cst{i,6})
                            obj_func = cst{i,6}{j};

                            target_dose = 0;
                            weight = 1;

                            if isprop(obj_func,'penalty') && ~isempty(obj_func.penalty)
                                weight = double(obj_func.penalty);
                            end

                            if isprop(obj_func,'parameters') && ~isempty(obj_func.parameters)
                                params = obj_func.parameters;

                                if isa(obj_func,'DoseConstraints.matRad_MinMaxDose')
                                    if numel(params) >= 2 && isnumeric(params{1}) && isnumeric(params{2})
                                        min_dose = double(params{1});
                                        max_dose = double(params{2});

                                        d_min(indices) = min_dose;
                                        d_max(indices) = max_dose;
                                        w_min(indices) = weight;
                                        w_max(indices) = weight;

                                        target_dose = (min_dose + max_dose) / 2;
                                        d_ref(indices) = target_dose;
                                        w(indices) = weight;
                                    end

                                elseif isa(obj_func,'DoseObjectives.matRad_MaxDVH')
                                    if numel(params) >= 1 && isnumeric(params{1})
                                        max_dose = double(params{1});
                                        d_max(indices) = max_dose;
                                        w_max(indices) = weight;
                                    end

                                elseif isa(obj_func,'DoseObjectives.matRad_MinDVH')
                                    if numel(params) >= 2 && isnumeric(params{1}) && isnumeric(params{2})
                                        min_dose = double(params{1});
                                        volume_percent = double(params{2});
                                        
                                        % For MinDVH, we want at least volume_percent to receive min_dose
                                        % This is equivalent to: D(100-volume_percent) >= min_dose
                                        % We enforce this by setting d_min for ALL voxels with very high penalty
                                        d_min(indices) = min_dose;
                                        w_min(indices) = weight * 1000; % Much higher penalty to act like hard constraint
                                    end

                                elseif isa(obj_func,'DoseObjectives.matRad_SquaredDeviation')
                                    if numel(params) >= 1 && isnumeric(params{1})
                                        target_dose = double(params{1});
                                        d_ref(indices) = target_dose;
                                        w(indices) = weight;
                                    end

                                elseif isa(obj_func,'DoseObjectives.matRad_EUD')
                                    if numel(params) >= 2 && isnumeric(params{1}) && isnumeric(params{2})
                                        target_eud = double(params{1});
                                        a_param = double(params{2});

                                        d_ref(indices) = target_eud;
                                        w(indices) = weight;
                                        eud_a(indices) = a_param;
                                        eud_mask(indices) = true;
                                    elseif numel(params) >= 1 && isnumeric(params{1})
                                        target_eud = double(params{1});
                                        a_param = 1.0;
                                        d_ref(indices) = target_eud;
                                        w(indices) = weight;
                                        eud_a(indices) = a_param;
                                        eud_mask(indices) = true;
                                    end

                                else
                                    if numel(params) >= 1 && isnumeric(params{1})
                                        target_dose = double(params{1});
                                        d_ref(indices) = target_dose;
                                        w(indices) = weight;
                                    end
                                end

                                if target_dose > 0 && ~isa(obj_func,'DoseConstraints.matRad_MinMaxDose') && ...
                                   ~isa(obj_func,'DoseObjectives.matRad_MaxDVH') && ~isa(obj_func,'DoseObjectives.matRad_EUD')
                                    d_ref(indices) = target_dose;
                                    w(indices) = weight;
                                end
                            end
                        end
                    else
                        matRad_cfg.dispInfo('Structure "%s" has no objectives\n', structName);
                    end
                end
            end

            obj.d_ref_cpu = d_ref;
            obj.w_cpu = w;
            obj.d_min_cpu = d_min;
            obj.d_max_cpu = d_max;
            obj.w_min_cpu = w_min;
            obj.w_max_cpu = w_max;
            obj.eud_a_cpu = eud_a;
            obj.eud_mask_cpu = eud_mask;

            % Precompute static masks once for speed
            obj.target_mask = obj.d_ref_cpu > 0;
            obj.min_mask    = obj.d_min_cpu > 0;
            obj.max_mask    = obj.d_max_cpu < inf;
            
            % Set prescription dose for hotspot control (max target dose including boosts)
            % Use the absolute maximum prescribed dose (e.g., GTV boost at 74 Gy, not PTV at 66 Gy)
            target_doses = d_ref(d_ref > 0);
            min_doses_active = d_min(d_min > 0);
            all_prescription_doses = [target_doses; min_doses_active];
            if ~isempty(all_prescription_doses)
                obj.prescriptionDose = max(all_prescription_doses);
                matRad_cfg.dispInfo('Prescription dose for hotspot control: %.2f Gy (max across all targets/boosts)\n', obj.prescriptionDose);
                matRad_cfg.dispInfo('Hotspot threshold: %.1f%% (%.2f Gy)\n', ...
                    obj.hotspotThreshold * 100, obj.prescriptionDose * obj.hotspotThreshold);
            end

            matRad_cfg.dispInfo('Final reference dose range: %.2f - %.2f\n', min(d_ref), max(d_ref));
            matRad_cfg.dispInfo('Final weight range: %.2e - %.2e\n', min(w), max(w));
            matRad_cfg.dispInfo('Voxels with target dose > 0: %d / %d\n', nnz(d_ref > 0), numel(d_ref));
            matRad_cfg.dispInfo('Voxels with min dose constraints: %d / %d\n', nnz(d_min > 0), numel(d_min));
            matRad_cfg.dispInfo('Voxels with max dose constraints: %d / %d\n', nnz(d_max < inf), numel(d_max));
            matRad_cfg.dispInfo('Voxels with EUD objectives: %d / %d\n', nnz(eud_mask), numel(eud_mask));

            target_voxels = 0; oar_voxels = 0;
            for i = 1:size(cst, 1)
                if ~isempty(cst{i,4}) && numel(cst{i,4}) >= 1 && ~isempty(cst{i,4}{1})
                    if size(cst,2) >= 6 && ~isempty(cst{i,6}) && numel(cst{i,6}) > 0
                        indices = cst{i,4}{1};
                        if any(d_ref(indices) > 0)
                            if strcmp(cst{i,3}, 'TARGET')
                                target_voxels = target_voxels + numel(indices);
                            else
                                oar_voxels = oar_voxels + numel(indices);
                            end
                        end
                    end
                end
            end
            matRad_cfg.dispInfo('Summary: %d target voxels, %d OAR voxels with objectives\n', target_voxels, oar_voxels);
        end

        function [statusmsg, statusflag] = GetStatus(obj)
            try
                switch obj.resultInfo.status
                    case 'converged'
                        statusmsg = 'Optimization converged successfully';
                        statusflag = 1;
                    case 'max_iterations'
                        statusmsg = 'Maximum number of iterations reached';
                        statusflag = 0;
                    case 'running'
                        statusmsg = 'Optimization is running';
                        statusflag = 0;
                    case 'no_objectives'
                        statusmsg = 'No optimization objectives found';
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
            % CPU optimizer is always available
            available = true;
        end
    end
end
