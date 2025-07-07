%% Example: Compare existing patient data with VMAT treatment plan
%
% This example loads an existing patient dataset, then creates a VMAT treatment
% plan for comparison with the original plan.
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
% Load ANON82686 patient data (same as in matRad_analyse_patient.m)
matRad_cfg = matRad_rc; %If this throws an error, run it from the parent directory first to set the paths
load('ANON82686.mat');

fprintf('Patient data loaded successfully!\n');

% Store a copy of the original plan results for comparison
if exist('resultGUI', 'var')
    original_resultGUI = resultGUI;
    fprintf('Original plan results stored for comparison.\n');
else
    error('No pre-calculated dose found in the dataset. Please run matRad_analyse_patient.m first.');
end

%% Configure VMAT treatment plan
fprintf('Configuring VMAT treatment plan...\n');

% Set meta parameters for treatment plan
pln = struct();
pln.radiationMode = 'photons';
pln.machine = 'Generic';
pln.numOfFractions = 30; % Match the original plan

% Biological model settings
pln.bioModel = 'none';   % Physical optimization
pln.multScen = 'nomScen'; % Use nominal scenario

% VMAT beam geometry settings
pln.propStf.bixelWidth = 5; % [mm]
pln.propStf.maxGantryAngleSpacing = 10;   % [°] fine spacing for dose calculation
pln.propStf.maxDAOGantryAngleSpacing = 20;  % [°] control points for DAO
pln.propStf.maxFMOGantryAngleSpacing = 60;  % [°] angles used in fluence optimization
pln.propStf.startingAngle = 0;           % [°]
pln.propStf.finishingAngle = 359;        % [°] full arc
pln.propStf.couchAngle = 0;              % [°]
pln.propStf.isoCenter = matRad_getIsoCenter(cst, ct, 0);
pln.propStf.generator = 'PhotonVMAT';
pln.propStf.continuousAperture = false;

% Dose calculation settings
pln.propDoseCalc.doseGrid.resolution.x = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.y = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.z = 3; % [mm]

% Sequencing settings
pln.propSeq.runSequencing = true;
pln.propSeq.sequencer = 'siochi';
pln.propSeq.numLevels = 7;

% Optimization settings
pln.propOpt.quantityOpt = 'physicalDose';
pln.propOpt.optimizer = 'IPOPT';
pln.propOpt.runDAO = true;
pln.propOpt.runVMAT = true;
pln.propOpt.preconditioner = true;

fprintf('VMAT plan configuration complete.\n');

%% Generate VMAT beam geometry
fprintf('Generating VMAT beam geometry...\n');
stf = matRad_generateStf(ct, cst, pln);
fprintf('Generated %d beams for VMAT plan.\n', numel(stf));

%% Dose calculation
fprintf('Calculating dose influence matrix...\n');
dij = matRad_calcDoseInfluence(ct, cst, stf, pln);
fprintf('Dose influence calculation complete.\n');

%% Fluence Optimization
fprintf('Running fluence optimization...\n');
vmatResultGUI = matRad_fluenceOptimization(dij, cst, pln, stf);
fprintf('Fluence optimization complete.\n');

%% Sequencing
fprintf('Running sequencing...\n');
vmatResultGUI = matRad_sequencing(vmatResultGUI, stf, dij, pln);
fprintf('Sequencing complete.\n');

%% Direct Aperture Optimization
fprintf('Running direct aperture optimization...\n');
vmatResultGUI = matRad_directApertureOptimization(dij, cst, vmatResultGUI.apertureInfo, vmatResultGUI, pln);
fprintf('Direct aperture optimization complete.\n');

%% Calculate quality indicators and DVH
fprintf('Calculating clinical quality indicators...\n');
vmatResultGUI = matRad_planAnalysis(vmatResultGUI, ct, cst, stf, pln);
fprintf('Quality indicators calculation complete.\n');

%% Calculate delivery metrics
fprintf('Calculating delivery metrics...\n');
vmatResultGUI = matRad_calcDeliveryMetrics(vmatResultGUI, pln, stf);
fprintf('Delivery metrics calculation complete.\n');

%% Visualization - Aperture shapes
fprintf('Visualizing aperture shapes...\n');
figure('Name', 'VMAT Aperture Shapes');
matRad_visApertureInfo(vmatResultGUI.apertureInfo);

%% Visualization - Dose Comparison
fprintf('Creating dose comparison visualizations...\n');

% Get center slice for visualization
centerSlice = round(size(ct.cubeDim, 3)/2);

% Set common dose window for comparison
maxDose = max(max(original_resultGUI.physicalDose(:)), max(vmatResultGUI.physicalDose(:)));
doseWindow = [0 maxDose];

% Original plan dose
figure('Name', 'Dose Comparison - Original vs VMAT');
subplot(2, 2, 1);
imagesc(squeeze(original_resultGUI.physicalDose(:,:,centerSlice)), doseWindow);
colormap(jet); colorbar;
title('Original Plan - Dose');
axis equal tight;

% VMAT plan dose
subplot(2, 2, 2);
imagesc(squeeze(vmatResultGUI.physicalDose(:,:,centerSlice)), doseWindow);
colormap(jet); colorbar;
title('VMAT Plan - Dose');
axis equal tight;

% Difference map
subplot(2, 2, 3);
diffDose = vmatResultGUI.physicalDose - original_resultGUI.physicalDose;
imagesc(squeeze(diffDose(:,:,centerSlice)));
colormap(jet); colorbar;
title('Dose Difference (VMAT - Original)');
axis equal tight;

% 3D dose visualization
subplot(2, 2, 4);
try
    matRad_visCtDose(vmatResultGUI, ct, cst);
    title('VMAT Dose Distribution');
catch
    warning('Unable to display 3D dose distribution.');
    imagesc(squeeze(vmatResultGUI.physicalDose(:,:,centerSlice)));
    colormap(jet); colorbar;
    title('VMAT Plan - Middle Slice');
    axis equal tight;
end

%% DVH Comparison
fprintf('Creating DVH comparison...\n');
figure('Name', 'DVH Comparison - Original vs VMAT');
hold on;

% Define colors for different structures
colors = lines(size(cst, 1));
lineStyles = {'-', '--'};  % Solid for original, dashed for VMAT

% Prepare for legend
legendEntries = cell(1, size(cst, 1) * 2);
dvhLines = [];
structureNames = {};

% Create common dose grid for both plans
maxDoseVal = max(max(original_resultGUI.physicalDose(:)), max(vmatResultGUI.physicalDose(:)));
commonDoseGrid = linspace(0, maxDoseVal * 1.1, 100);

% Store all volume points for both plans
original_volumePoints = cell(size(cst, 1), 1);
vmat_volumePoints = cell(size(cst, 1), 1);

% Plot DVH for each structure - both plans
legendIdx = 1;
for i = 1:size(cst, 1)
    if ~isempty(cst{i,4}) && ~isempty(cst{i,4}{1})
        structName = cst{i,2};
        structureNames{end+1} = structName;
        
        % Original plan DVH
        originalVoxels = cst{i,4}{1};
        originalDoses = original_resultGUI.physicalDose(originalVoxels);
        original_volumePoints{i} = zeros(size(commonDoseGrid));
        
        % Calculate DVH points for original plan
        for j = 1:length(commonDoseGrid)
            original_volumePoints{i}(j) = 100 * sum(originalDoses >= commonDoseGrid(j)) / length(originalVoxels);
        end
        
        % VMAT plan DVH
        vmatDoses = vmatResultGUI.physicalDose(originalVoxels);
        vmat_volumePoints{i} = zeros(size(commonDoseGrid));
        
        % Calculate DVH points for VMAT plan
        for j = 1:length(commonDoseGrid)
            vmat_volumePoints{i}(j) = 100 * sum(vmatDoses >= commonDoseGrid(j)) / length(originalVoxels);
        end
        
        % Plot original plan DVH
        h1 = plot(commonDoseGrid, original_volumePoints{i}, 'LineWidth', 2, 'Color', colors(i,:), 'LineStyle', lineStyles{1});
        dvhLines(end+1) = h1;
        legendEntries{legendIdx} = [structName ' - Original'];
        legendIdx = legendIdx + 1;
        
        % Plot VMAT plan DVH
        h2 = plot(commonDoseGrid, vmat_volumePoints{i}, 'LineWidth', 2, 'Color', colors(i,:), 'LineStyle', lineStyles{2});
        dvhLines(end+1) = h2;
        legendEntries{legendIdx} = [structName ' - VMAT'];
        legendIdx = legendIdx + 1;
    end
end

% Add data cursor functionality
dcm = datacursormode(gcf);
dcm.Enable = 'on';
dcm.UpdateFcn = @(obj, event_obj) dvhDataCursorUpdateFcn(obj, event_obj, dvhLines, structureNames, lineStyles);

% Add legend, grid and labels
if legendIdx > 1
    legend(legendEntries(1:legendIdx-1), 'Location', 'northeast', 'FontSize', 10);
end

grid on;
xlabel('Dose [Gy]', 'FontSize', 12);
ylabel('Volume [%]', 'FontSize', 12);
title('DVH Comparison: Original vs VMAT Plan', 'FontSize', 14);
hold off;

%% Quality Indicator Comparison
fprintf('Comparing plan quality metrics...\n');

% Create a table to compare quality metrics between plans
qiNames = {'D_95', 'D_98', 'D_mean', 'D_max', 'HI', 'CI'};
qiTable = table();

% Populate the table with metrics for each structure
for i = 1:size(cst, 1)
    if ~isempty(cst{i,4}) && ~isempty(cst{i,4}{1})
        structName = cst{i,2};
        
        for j = 1:length(qiNames)
            metricName = qiNames{j};
            
            % Get metric for original plan if available
            if isfield(original_resultGUI, 'qi') && isfield(original_resultGUI.qi(i), metricName)
                originalVal = original_resultGUI.qi(i).(metricName);
            else
                originalVal = NaN;
            end
            
            % Get metric for VMAT plan if available
            if isfield(vmatResultGUI, 'qi') && isfield(vmatResultGUI.qi(i), metricName)
                vmatVal = vmatResultGUI.qi(i).(metricName);
            else
                vmatVal = NaN;
            end
            
            % Add to table
            rowName = [structName '_' metricName];
            qiTable.(rowName) = [originalVal; vmatVal];
        end
    end
end

% Set row names
qiTable.Properties.RowNames = {'Original'; 'VMAT'};

% Display the table
disp('Quality Metrics Comparison:');
disp(qiTable);

%% Delivery Metrics Comparison
if isfield(vmatResultGUI, 'deliveryMetrics')
    disp('VMAT Delivery Metrics:');
    disp(vmatResultGUI.deliveryMetrics);
end

%% Function to handle data cursor updates for DVH
function txt = dvhDataCursorUpdateFcn(~, event_obj, dvhLines, structureNames, lineStyles)
    % Get the line that was clicked
    clickedLine = get(event_obj, 'Target');
    
    % Find which structure and plan this line represents
    lineIdx = find(dvhLines == clickedLine, 1);
    
    if ~isempty(lineIdx)
        structIdx = ceil(lineIdx/2);
        if structIdx <= length(structureNames)
            structureName = structureNames{structIdx};
        else
            structureName = 'Unknown Structure';
        end
        
        % Determine if this is original or VMAT plan
        if mod(lineIdx, 2) == 1
            planName = 'Original Plan';
        else
            planName = 'VMAT Plan';
        end
    else
        structureName = 'Unknown Structure';
        planName = 'Unknown Plan';
    end
    
    % Get position data
    pos = get(event_obj, 'Position');
    
    % Create the text for the tooltip
    txt = {['Structure: ', structureName], ...
           ['Plan: ', planName], ...
           ['Dose: ', num2str(pos(1), '%.2f'), ' Gy'], ...
           ['Volume: ', num2str(pos(2), '%.2f'), ' %']};
end

fprintf('\nComparison between original plan and VMAT plan complete!\n');
