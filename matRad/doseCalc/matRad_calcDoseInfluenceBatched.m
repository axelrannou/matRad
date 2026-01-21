function dij = matRad_calcDoseInfluenceBatched(ct, cst, stf, pln, batchSize, options)
% matRad_calcDoseInfluenceBatched
% Calculate dij in batches and build sparse matrix incrementally
% Saves intermediate results to disk to allow resuming after crash
% 
% Input:
%   ct, cst, stf, pln - standard matRad inputs
%   batchSize - number of beams to process at once (default: 10)
%   options - struct with optional fields:
%       .useSinglePrecision - use single precision to halve memory (default: false)
%       .doseThreshold      - relative threshold (0-1), discard values below 
%                             this fraction of max per beamlet (default: 0)
%       .absoluteThreshold  - absolute threshold in Gy, discard values below this (default: 0)
%
% Output:
%   dij - structure with physicalDose as sparse matrix

    if nargin < 5 || isempty(batchSize)
        batchSize = 10;
    end
    
    if nargin < 6
        options = struct();
    end
    
    % Set default options
    if ~isfield(options, 'useSinglePrecision')
        options.useSinglePrecision = false;
    end
    if ~isfield(options, 'doseThreshold')
        options.doseThreshold = 0;
    end
    if ~isfield(options, 'absoluteThreshold')
        options.absoluteThreshold = 0;
    end
    
    matRad_cfg = MatRad_Config.instance();
    
    numBeams = length(stf);
    matRad_cfg.dispInfo('Calculating dij in batches of %d beams (%d total)\n', batchSize, numBeams);
    matRad_cfg.dispInfo('Options:\n');
    if options.useSinglePrecision
        matRad_cfg.dispInfo('  - Single precision: enabled\n');
    end
    if options.doseThreshold > 0
        matRad_cfg.dispInfo('  - Relative dose threshold: %.2f%% of beamlet max\n', options.doseThreshold * 100);
    end
    if options.absoluteThreshold > 0
        matRad_cfg.dispInfo('  - Absolute dose threshold: %.2e Gy\n', options.absoluteThreshold);
    end
    
    % Setup cache directory
    cacheDir = fullfile(pwd, 'dij_cache');
    if ~exist(cacheDir, 'dir')
        mkdir(cacheDir);
    end
    
    % Incremental sparse matrix file
    incrementalFile = fullfile(cacheDir, 'dij_incremental.mat');
    progressFile = fullfile(cacheDir, 'dij_progress.mat');
    
    % First pass: calculate dimensions by processing first beam
    matRad_cfg.dispInfo('Determining matrix dimensions...\n');
    stf_first = stf(1);
    dij_first = matRad_calcDoseInfluence(ct, cst, stf_first, pln);
    
    if iscell(dij_first.physicalDose)
        D_first = dij_first.physicalDose{1};
    else
        D_first = dij_first.physicalDose;
    end
    
    numVoxels = size(D_first, 1);
    
    % Count total beamlets
    totalBeamlets = 0;
    beamletOffset = zeros(numBeams + 1, 1);
    beamletOffset(1) = 1;
    
    for b = 1:numBeams
        numRays = stf(b).numOfRays;
        totalBixels = 0;
        for r = 1:numRays
            totalBixels = totalBixels + stf(b).numOfBixelsPerRay(r);
        end
        totalBeamlets = totalBeamlets + totalBixels;
        beamletOffset(b + 1) = beamletOffset(b) + totalBixels;
    end
    
    matRad_cfg.dispInfo('Total voxels: %d, Total beamlets: %d\n', numVoxels, totalBeamlets);
    
    clear dij_first D_first;
    
    % Check if we can resume from previous progress
    startBatch = 1;
    currentBeamlet = 1;
    
    if exist(progressFile, 'file') && exist(incrementalFile, 'file')
        matRad_cfg.dispInfo('Found previous progress, attempting to resume...\n');
        try
            prog = load(progressFile);
            if prog.numBeams == numBeams && prog.totalBeamlets == totalBeamlets
                startBatch = prog.lastCompletedBatch + 1;
                currentBeamlet = prog.currentBeamlet;
                matRad_cfg.dispInfo('Resuming from batch %d (beamlet %d)\n', startBatch, currentBeamlet);
            else
                matRad_cfg.dispInfo('Configuration changed, starting fresh\n');
                delete(incrementalFile);
                delete(progressFile);
            end
        catch
            matRad_cfg.dispInfo('Could not load progress, starting fresh\n');
            if exist(incrementalFile, 'file'), delete(incrementalFile); end
            if exist(progressFile, 'file'), delete(progressFile); end
        end
    end
    
    % Initialize or load sparse matrix
    if startBatch == 1
        D_full = sparse(numVoxels, totalBeamlets);
        matRad_cfg.dispInfo('Initialized empty sparse matrix\n');
    else
        matRad_cfg.dispInfo('Loading existing sparse matrix...\n');
        loaded = load(incrementalFile, 'D_full');
        D_full = loaded.D_full;
        clear loaded;
        matRad_cfg.dispInfo('Loaded matrix with %d non-zeros\n', nnz(D_full));
    end
    
    % Process beams in batches
    totalBatches = ceil(numBeams / batchSize);
    
    for batchNum = startBatch:totalBatches
        batchStart = (batchNum - 1) * batchSize + 1;
        batchEnd = min(batchNum * batchSize, numBeams);
        batchBeams = batchStart:batchEnd;
        
        matRad_cfg.dispInfo('\n');
        matRad_cfg.dispInfo('============================================================\n');
        matRad_cfg.dispInfo('BATCH %d/%d: Processing beams %d-%d of %d\n', ...
            batchNum, totalBatches, batchStart, batchEnd, numBeams);
        matRad_cfg.dispInfo('============================================================\n');
        
        % Calculate beamlet offset for this batch
        if batchNum > 1 || startBatch > 1
            currentBeamlet = beamletOffset(batchStart);
        end
        
        % Extract stf for this batch
        stf_batch = stf(batchBeams);
        
        % Calculate dij for batch
        tic;
        dij_batch = matRad_calcDoseInfluence(ct, cst, stf_batch, pln);
        batchTime = toc;
        
        if iscell(dij_batch.physicalDose)
            D_batch = dij_batch.physicalDose{1};
        else
            D_batch = dij_batch.physicalDose;
        end
        
        % Get batch dimensions
        [batchRows, batchCols, batchVals] = find(D_batch);
        originalNnz = length(batchVals);
        
        % Apply relative threshold per column (beamlet) - VECTORIZED for speed
        if options.doseThreshold > 0
            % Use accumarray to find max per column in one operation
            numCols = size(D_batch, 2);
            colMaxVals = accumarray(batchCols, abs(batchVals), [numCols, 1], @max);
            
            % Calculate threshold for each value based on its column's max
            thresholds = colMaxVals(batchCols) * options.doseThreshold;
            
            % Apply threshold in one vectorized operation
            keepMask = abs(batchVals) >= thresholds;
            batchRows = batchRows(keepMask);
            batchCols = batchCols(keepMask);
            batchVals = batchVals(keepMask);
        end
        
        % Apply absolute threshold
        if options.absoluteThreshold > 0
            keepMask = abs(batchVals) >= options.absoluteThreshold;
            batchRows = batchRows(keepMask);
            batchCols = batchCols(keepMask);
            batchVals = batchVals(keepMask);
        end
        
        % Report reduction
        if options.doseThreshold > 0 || options.absoluteThreshold > 0
            finalNnz = length(batchVals);
            matRad_cfg.dispInfo('  Thresholding: kept %d/%d non-zeros (%.1f%% reduction)\n', ...
                finalNnz, originalNnz, 100*(1 - finalNnz/max(1,originalNnz)));
        end
        
        batchCols = batchCols + currentBeamlet - 1;  % Adjust to global column index
        
        % Convert to single precision if enabled
        if options.useSinglePrecision
            batchVals = single(batchVals);
        end
        
        % Create sparse matrix for this batch and add to full matrix
        D_batch_global = sparse(double(batchRows), double(batchCols), double(batchVals), numVoxels, totalBeamlets);
        D_full = D_full + D_batch_global;
        
        numNonZeros = length(batchRows);
        currentBeamlet = currentBeamlet + size(D_batch, 2);
        
        matRad_cfg.dispInfo('  Batch took %.1f s, added %d non-zeros (total: %d)\n', ...
            batchTime, numNonZeros, nnz(D_full));
        
        % Clear batch data
        clear dij_batch D_batch D_batch_global batchRows batchCols batchVals;
        
        % Save progress to disk after each batch
        matRad_cfg.dispInfo('  Saving progress to disk...\n');
        save(incrementalFile, 'D_full', '-v7.3');
        
        % Save progress marker
        lastCompletedBatch = batchNum;
        save(progressFile, 'lastCompletedBatch', 'currentBeamlet', 'numBeams', 'totalBeamlets', 'numVoxels', 'beamletOffset');
        
        matRad_cfg.dispInfo('  Progress saved (can resume if interrupted)\n');
        
        % Force garbage collection
        java.lang.System.gc();
    end
    
    matRad_cfg.dispInfo('\nAll batches complete! Final matrix: %d x %d with %d non-zeros (%.2f GB)\n', ...
        size(D_full, 1), size(D_full, 2), nnz(D_full), nnz(D_full) * 16 / 1e9);
    
    % Get a reference dij to copy all necessary fields
    matRad_cfg.dispInfo('Getting reference dij structure for metadata...\n');
    stf_ref = stf(1);
    dij_ref = matRad_calcDoseInfluence(ct, cst, stf_ref, pln);
    
    % Create output structure starting from reference
    dij = struct();
    
    % Copy all fields from reference dij
    refFields = fieldnames(dij_ref);
    for f = 1:length(refFields)
        fieldName = refFields{f};
        if ~strcmp(fieldName, 'physicalDose')
            dij.(fieldName) = dij_ref.(fieldName);
        end
    end
    
    % Set our computed physicalDose
    dij.physicalDose = {D_full};
    
    % Update fields that depend on full beam set
    dij.numOfBeams = numBeams;
    dij.totalNumOfBixels = totalBeamlets;
    dij.numOfVoxels = numVoxels;
    
    % Build beamNum vector
    beamNum = zeros(totalBeamlets, 1);
    for b = 1:numBeams
        startIdx = beamletOffset(b);
        endIdx = beamletOffset(b + 1) - 1;
        beamNum(startIdx:endIdx) = b;
    end
    dij.beamNum = beamNum;
    
    % Build bixelNum vector
    bixelNum = zeros(totalBeamlets, 1);
    for b = 1:numBeams
        startIdx = beamletOffset(b);
        endIdx = beamletOffset(b + 1) - 1;
        numBixelsInBeam = endIdx - startIdx + 1;
        bixelNum(startIdx:endIdx) = 1:numBixelsInBeam;
    end
    dij.bixelNum = bixelNum;
    
    % Build rayNum vector
    rayNum = zeros(totalBeamlets, 1);
    globalBixelIdx = 1;
    for b = 1:numBeams
        for r = 1:stf(b).numOfRays
            numBixelsInRay = stf(b).numOfBixelsPerRay(r);
            rayNum(globalBixelIdx:globalBixelIdx + numBixelsInRay - 1) = r;
            globalBixelIdx = globalBixelIdx + numBixelsInRay;
        end
    end
    dij.rayNum = rayNum;
    
    clear dij_ref D_full;
    
    % Clean up progress files (keep incremental for potential reuse)
    if exist(progressFile, 'file')
        delete(progressFile);
    end
    
    matRad_cfg.dispInfo('Done!\n');
end
