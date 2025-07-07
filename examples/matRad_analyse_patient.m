%% Example: Load and analyze ANON82686 patient data and generate DVH
%
% This example loads a patient dataset, computes dose influence matrix,
% optimizes the treatment plan, and generates a dose-volume histogram (DVH).
%
% Copyright 2023 the matRad development team. 
%
% This file is part of the matRad project. It is subject to the license 
% terms in the LICENSE file found in the top-level directory of this 
% distribution and at https://github.com/e0404/matRad/LICENSES.txt. No part 
% of the matRad project, including this file, may be copied, modified, 
% propagated, or distributed except according to the terms contained in the 
% LICENSE file.
%
% ========================================================================= %

%% Set up environment
clc, clear, close all;

% Add path if running this in a standalone folder
addpath(genpath('..'));

%% Load patient data
% Load ANON82686 patient data
matRad_cfg = matRad_rc; %If this throws an error, run it from the parent directory first to set the paths
%load('jane_doe.mat');
load('john_doe.mat');

fprintf('Patient data loaded successfully!\n');

% Display patient information
fprintf('Number of CT slices: %d\n', size(ct.cubeDim, 3));
fprintf('Number of VOIs: %d\n', size(cst, 1));

%% Treatment Plan
% The next step is to define your treatment plan labeled as 'pln'. This 
% structure requires input from the treatment planner and defines 
% the most important cornerstones of your treatment plan.

pln.radiationMode   = 'photons';   % either photons / protons / carbon
pln.machine         = 'Generic';
pln.numOfFractions  = 30;
 
pln.propStf.gantryAngles    = [0:40:359];
pln.propStf.couchAngles     = zeros(1,numel(pln.propStf.gantryAngles));
pln.propStf.bixelWidth      = 5;
pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
pln.propStf.isoCenter       = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);

pln.bioModel = 'none'; 
pln.multScen = 'nomScen';
pln.propStf.continuousAperture  = false;

% dose calculation settings
pln.propDoseCalc.doseGrid.resolution.x = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.y = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.z = 3; % [mm]

% We can also use other solver for optimization than IPOPT. matRad 
% currently supports fmincon from the MATLAB Optimization Toolbox. First we
% check if the fmincon-Solver is available, and if it es, we set in in the
% pln.propOpt.optimizer vairable. Otherwise wie set to the default
% optimizer 'IPOPT'
if matRad_OptimizerFmincon.IsAvailable()
    pln.propOpt.optimizer = 'fmincon';   
else
    pln.propOpt.optimizer = 'IPOPT';
end
pln.propOpt.quantityOpt = 'physicalDose';  

%%
% Enable sequencing and direct aperture optimization (DAO).
pln.propSeq.runSequencing = true;


%% Skip dose calculation and optimization - using existing dose
fprintf('Using existing dose distribution...\n');

% Check if dose exists in the loaded data
if ~exist('resultGUI', 'var')
    error('No pre-calculated dose found in the dataset. Please ensure resultGUI variable is loaded.');
end

fprintf('Pre-calculated dose found!\n');

%% Calculate quality indicators
fprintf('Calculating clinical quality indicators...\n');

% Using the newer matRad_planAnalysis directly instead of matRad_indicatorWrapper
% Create an empty steering file structure if not available
if ~exist('stf', 'var')
    stf = struct();
end

% Use the newer plan analysis function to avoid the numOfCtScen error
% resultGUI = matRad_planAnalysis(resultGUI, ct, cst, stf, pln);

% Calculate DVH directly using a more efficient approach
fprintf('Calculating DVH directly...\n');

% Create a simple DVH structure
dvh = struct();
dvh.doseGrid = linspace(0, max(resultGUI.physicalDose(:))*1.1, 100); % Use fewer dose bins
dvh.volumePoints = cell(size(cst, 1), 1);

% Calculate simple DVH for each structure with more efficient implementation
for i = 1:size(cst, 1)
    dvh.volumePoints{i} = [];
    
    % Only proceed for structures with voxel indices
    if ~isempty(cst{i,4}) && ~isempty(cst{i,4}{1})
        fprintf('Processing structure %d: %s...\n', i, cst{i,2});
        
        % Get structure voxel indices
        voxelIndices = cst{i,4}{1};
        
        % Process in chunks if needed to avoid integer overflow
        chunkSize = 1000000; % Process in manageable chunks
        numChunks = ceil(length(voxelIndices) / chunkSize);
        
        % Pre-allocate histogram result
        histResult = zeros(size(dvh.doseGrid));
        
        % Process each chunk
        for chunk = 1:numChunks
            % Get indices for this chunk
            startIdx = (chunk-1)*chunkSize + 1;
            endIdx = min(chunk*chunkSize, length(voxelIndices));
            chunkIndices = voxelIndices(startIdx:endIdx);
            
            % Extract doses for these voxels
            doseValues = resultGUI.physicalDose(chunkIndices);
            
            % Calculate histogram for this chunk
            for doseLevel = 1:length(dvh.doseGrid)
                histResult(doseLevel) = histResult(doseLevel) + sum(doseValues >= dvh.doseGrid(doseLevel));
            end
        end
        
        % Convert to percentage
        dvh.volumePoints{i} = 100 * histResult / length(voxelIndices);
    end
end

fprintf('DVH calculation complete!\n');

%% Visualization
% Visualize dose distribution
fprintf('Creating visualizations...\n');
try
    % Try to use matRad_visual.m instead of matRad_plotSlice
    figure('Name', 'Dose distribution');
    matRad_visCtDose(resultGUI, ct, cst);
    title('Dose distribution');
catch
    % Fallback visualization method if matRad_visCtDose is not available
    try
        % Simple alternative visualization - just show the middle slice
        middleSlice = round(size(resultGUI.physicalDose, 3)/2);
        if isempty(middleSlice) || middleSlice < 1
            middleSlice = 1;
        end
        
        imagesc(squeeze(resultGUI.physicalDose(:,:,middleSlice)));
        colormap('jet');
        colorbar;
        title('Dose distribution (Middle Slice)');
        axis equal tight;
    catch
        warning('Unable to display dose distribution. Skipping visualization.');
    end
end

% Plot DVH using a custom function since matRad_plotDVH might not be available
fprintf('Creating custom DVH plot...\n');
figure('Name', 'DVH');
hold on;

% Define a color map for different structures
colors = lines(size(cst, 1));

% Store plotted lines to use with data cursor
dvhLines = [];
structureNames = {};

% Simplified approach - assume we have a valid dvh struct or create one
try
    % Check if dvh is a valid structure with required fields
    if ~isfield(dvh, 'doseGrid') || ~isfield(dvh, 'volumePoints')
        error('DVH structure is missing required fields');
    end
    
    % Plot each structure separately with labels
    legendEntries = {};
    for i = 1:size(cst, 1)
        try
            % Safely access volume points
            if i <= numel(dvh.volumePoints) && ~isempty(dvh.volumePoints{i})
                h = plot(dvh.doseGrid, dvh.volumePoints{i}, 'LineWidth', 2, 'Color', colors(i,:));
                dvhLines(end+1) = h;
                structureNames{end+1} = cst{i,2};
                legendEntries{end+1} = cst{i,2};
            end
        catch
            % Skip on error
            fprintf('Skipping structure %d due to error\n', i);
        end
    end
catch
    % Fallback to completely manual calculation
    fprintf('Using manual DVH calculation...\n');
    legendEntries = {};
    
    % For each structure, manually calculate DVH
    for i = 1:size(cst, 1)
        try
            % Only calculate for structures with indices
            if ~isempty(cst{i,4})
                voxelIndices = cst{i,4}{1};
                if ~isempty(voxelIndices)
                    % Get dose values for this structure
                    structDoses = resultGUI.physicalDose(voxelIndices);
                    
                    % Create dose bins
                    doseBins = linspace(0, max(resultGUI.physicalDose(:)), 100);
                    volumePoints = zeros(size(doseBins));
                    
                    % Calculate cumulative histogram
                    for j = 1:length(doseBins)
                        volumePoints(j) = 100 * sum(structDoses >= doseBins(j)) / length(voxelIndices);
                    end
                    
                    % Plot the DVH
                    h = plot(doseBins, volumePoints, 'LineWidth', 2, 'Color', colors(i,:));
                    dvhLines(end+1) = h;
                    structureNames{end+1} = cst{i,2};
                    legendEntries{end+1} = cst{i,2};
                end
            end
        catch
            % Skip on error
            fprintf('Skipping structure %d in manual calculation\n', i);
        end
    end
end

% Add data cursor functionality to show structure names on hover
dcm = datacursormode(gcf);
dcm.Enable = 'on';
dcm.UpdateFcn = @(obj, event_obj) dvhDataCursorUpdateFcn(obj, event_obj, dvhLines, structureNames);

% Add legend if we have structures to show
if exist('legendEntries', 'var') && ~isempty(legendEntries)
    try
        legend(legendEntries, 'Location', 'northeast', 'FontSize', 10);
    catch
        warning('Could not create legend');
    end
end

% Add grid and labels
grid on;
xlabel('Dose [Gy]', 'FontSize', 12);
ylabel('Volume [%]', 'FontSize', 12);
title('Dose-Volume Histogram', 'FontSize', 14);
hold off;

% Function to handle data cursor updates
function txt = dvhDataCursorUpdateFcn(~, event_obj, dvhLines, structureNames)
    % Get the line that was clicked
    clickedLine = get(event_obj, 'Target');
    
    % Find which structure this line represents
    lineIdx = find(dvhLines == clickedLine, 1);
    
    if ~isempty(lineIdx) && lineIdx <= length(structureNames)
        structureName = structureNames{lineIdx};
    else
        structureName = 'Unknown Structure';
    end
    
    % Get position data
    pos = get(event_obj, 'Position');
    
    % Create the text for the tooltip
    txt = {['Structure: ', structureName], ...
           ['Dose: ', num2str(pos(1), '%.2f'), ' Gy'], ...
           ['Volume: ', num2str(pos(2), '%.2f'), ' %']};
end

%% Indicator Calculation and display of DVH and QI
resultGUI = matRad_planAnalysis(resultGUI,ct,cst,stf,pln);

% Display summary of clinical indicators if available
if exist('qi', 'var') && isstruct(qi)
    fprintf('\nClinical Quality Indicators Summary:\n');
    fprintf('-----------------------------------\n');
    fields = fieldnames(qi);
    for i = 1:numel(fields)
        if isnumeric(qi.(fields{i}))
            fprintf('%s: %.2f\n', fields{i}, qi.(fields{i}));
        end
    end
end

fprintf('\nPatient analysis complete!\n');
