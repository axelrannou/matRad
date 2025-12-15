classdef matRad_OptimizerPGD_CPU < matRad_Optimizer
% matRad_OptimizerPGD_CPU: CPU Projected Gradient Descent optimizer
%
% CPU version mirroring the objective/gradient handling from the GPU PGD.

    properties
        options = struct();

        % PGD parameters
        stepSize = 0.1;
        adaptiveStepSize = true;
        stepSizeReduction = 0.5;
        minStepSize = 1e-10;
        maxFluence = 10.0;

        % Convergence
        relTolGrad = 1e-4;
        relTolObj = 1e-6;
        maxIter = 5000;

        % CPU storage
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

        % Speed/compute controls
        useSinglePrecision = false;   % set true to run in single
        progressEvery = 10;           % print every N iterations

        % Precomputed masks (static across iterations)
        target_mask = [];
        min_mask = [];
        max_mask = [];
    end

    properties (SetAccess = protected)
        wResult
        resultInfo
    end

    methods
        function obj = matRad_OptimizerPGD_CPU()
            obj@matRad_Optimizer();
            obj.wResult = [];
            obj.resultInfo = [];
        end

        function obj = optimize(obj, w0, optiProb, dij, cst)
            matRad_cfg = MatRad_Config.instance();

            % Optional user step size
            if isfield(optiProb, 'stepSize') && ~isempty(optiProb.stepSize)
                obj.stepSize = optiProb.stepSize;
            elseif isfield(optiProb, 'propOpt') && isfield(optiProb.propOpt, 'stepSize') && ~isempty(optiProb.propOpt.stepSize)
                obj.stepSize = optiProb.propOpt.stepSize;
            end

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

            % Initialize
            x = obj.x_cpu;
            step_size = obj.stepSize;

            % Precompute bounds once (adjust if bounds change per-iter)
            lb = optiProb.lowerBounds(x);
            ub = optiProb.upperBounds(x);

            % History
            obj.resultInfo.objectiveHistory = zeros(obj.maxIter, 1);

            matRad_cfg.dispInfo('Starting CPU PGD optimization...\n');

            for iter = 1:obj.maxIter
                % Objective and gradient on CPU
                [obj_val, grad] = obj.computeObjectiveAndGradientCPU(x);

                % Store
                if iter == 1
                    obj.resultInfo.initialObjective = obj_val;
                end
                obj.resultInfo.objectiveHistory(iter) = obj_val;

                % Convergence by relative objective change
                if iter > 1
                    prev = obj.resultInfo.objectiveHistory(iter-1);
                    if prev ~= 0
                        rel_obj_change = abs(obj_val - prev) / abs(prev);
                        if rel_obj_change < obj.relTolObj
                            matRad_cfg.dispInfo('Converged: relative objective change < %.2e\n', obj.relTolObj);
                            obj.resultInfo.status = 'converged';
                            break;
                        end
                    end
                end

                % Early abort if stuck at zero
                if iter > 10 && obj.resultInfo.objectiveHistory(iter) == 0
                    matRad_cfg.dispWarning('Objective stuck at zero - likely no meaningful targets\n');
                    obj.resultInfo.status = 'stuck_at_zero';
                    break;
                end

                % Adaptive step size (monotonic backtracking)
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

                % PGD step + projection
                x = x - step_size .* grad;
                x = max(lb, min(ub, x));

                % Progress
                if mod(iter, obj.progressEvery) == 0 || iter == 1
                    improvement = 0;
                    if iter > 1 && obj.resultInfo.objectiveHistory(1) > 0
                        improvement = 100 * (obj.resultInfo.objectiveHistory(1) - obj.resultInfo.objectiveHistory(iter)) / obj.resultInfo.objectiveHistory(1);
                    end
                    grad_norm = norm(grad);
                    matRad_cfg.dispInfo('Iter %3d: Obj = %.6e, Step = %.2e, GradNorm = %.2e, Improvement = %.2f%% [CPU]\n', ...
                        iter, obj.resultInfo.objectiveHistory(iter), step_size, grad_norm, improvement);
                end
            end

            % Finalize with timing
            obj.resultInfo.optimizationTime = toc(optimization_start_time);
            obj.wResult = x;
            obj.resultInfo.iterations = iter;
            obj.resultInfo.finalObjective = obj.resultInfo.objectiveHistory(iter);
            obj.resultInfo.objectiveHistory = obj.resultInfo.objectiveHistory(1:iter);

            if ~ismember(obj.resultInfo.status, {'converged','step_size_too_small','stuck_at_zero'})
                obj.resultInfo.status = 'max_iterations';
            end
            
            % Format time display
            total_seconds = obj.resultInfo.optimizationTime;
            hours = floor(total_seconds / 3600);
            minutes = floor(mod(total_seconds, 3600) / 60);
            seconds = mod(total_seconds, 60);
            
            matRad_cfg.dispInfo('CPU optimization completed after %d iterations in %dh %dm %.2fs\n', ...
                iter, hours, minutes, seconds);
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

            % 2) Minimum dose constraints
            if any(obj.min_mask)
                idx = obj.min_mask;
                underdose = obj.d_min_cpu(idx) - dose(idx);
                active = underdose > 0;
                if any(active)
                    u_idx = find(idx); u_idx = u_idx(active);
                    u_val = underdose(active);
                    wv = obj.w_min_cpu(u_idx);
                    obj_val = obj_val + 0.5 * sum(wv .* (u_val .^ 2));
                    % negative sign (dose below min)
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
            % CPU optimizer is always available
            available = true;
        end
    end
end