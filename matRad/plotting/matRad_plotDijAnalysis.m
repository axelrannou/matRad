% filepath: /home/arannou/Documents/Workspace/matrad_env/matRad/matRad/plotting/matRad_plotDijAnalysis.m
function matRad_plotDijAnalysis(dij, figureTitle)
% matRad_plotDijAnalysis
%   Analyzes and visualizes the beamlet weight distribution in a dij structure
%
% call
%   matRad_plotDijAnalysis(dij)
%   matRad_plotDijAnalysis(dij, figureTitle)
%
% input
%   dij:          matRad dij struct containing dose influence matrix
%   figureTitle:  (optional) custom title for the figure window
%
% output
%   Creates a figure with 6 subplots showing various aspects of the
%   beamlet weight distribution
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright 2024 the matRad development team.
%
% This file is part of the matRad project. It is subject to the license
% terms in the LICENSE file found in the top-level directory of this
% distribution and at https://github.com/e0404/matRad/LICENSE.md. No part
% of the matRad project, including this file, may be copied, modified,
% propagated, or distributed except according to the terms contained in the
% LICENSE file.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matRad_cfg = MatRad_Config.instance();

% Set default figure title
if nargin < 2 || isempty(figureTitle)
    figureTitle = 'Dij Weight Distribution Analysis';
end

matRad_cfg.dispInfo('\n=== Analyzing dij beamlet weight distribution ===\n');

% Extract the dose influence matrix
if iscell(dij.physicalDose)
    D = dij.physicalDose{1};
else
    D = dij.physicalDose;
end

% Get all non-zero weights
[~, ~, weights] = find(D);
numNonZero = length(weights);
absWeights = abs(weights);

% Calculate and display statistics
matRad_cfg.dispInfo('Total beamlets: %d\n', size(D, 2));
matRad_cfg.dispInfo('Total voxels: %d\n', size(D, 1));
matRad_cfg.dispInfo('Non-zero entries: %d (%.2f%% sparse)\n', ...
    numNonZero, 100*(1 - numNonZero/(size(D,1)*size(D,2))));
matRad_cfg.dispInfo('Weight statistics:\n');
matRad_cfg.dispInfo('  Min:    %.6e Gy\n', min(absWeights));
matRad_cfg.dispInfo('  Max:    %.6e Gy\n', max(absWeights));
matRad_cfg.dispInfo('  Mean:   %.6e Gy\n', mean(absWeights));
matRad_cfg.dispInfo('  Median: %.6e Gy\n', median(absWeights));
matRad_cfg.dispInfo('  Std:    %.6e Gy\n', std(absWeights));

% Create figure with multiple subplots
figure('Name', figureTitle, 'Position', [100, 100, 1400, 900]);

% 1. Histogram of all weights
subplot(2, 3, 1);
histogram(absWeights, 100, 'EdgeColor', 'none', 'FaceColor', [0.2 0.5 0.8]);
xlabel('Absolute Weight [Gy]', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('Distribution of All Non-Zero Weights', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% 2. Log-scale histogram
subplot(2, 3, 2);
histogram(log10(absWeights), 100, 'EdgeColor', 'none', 'FaceColor', [0.8 0.4 0.2]);
xlabel('log_{10}(Absolute Weight) [Gy]', 'FontSize', 11);
ylabel('Count', 'FontSize', 11);
title('Log-Scale Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% 3. Cumulative distribution
subplot(2, 3, 3);
sortedWeights = sort(absWeights, 'descend');
cumulativePercent = (1:numNonZero) / numNonZero * 100;
semilogx(sortedWeights, cumulativePercent, 'LineWidth', 2, 'Color', [0.3 0.7 0.3]);
xlabel('Absolute Weight [Gy]', 'FontSize', 11);
ylabel('Cumulative Percentage (%)', 'FontSize', 11);
title('Cumulative Distribution (largest to smallest)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;
xlim([min(absWeights) max(absWeights)]);
ylim([0 100]);

% 4. Percentile analysis
subplot(2, 3, 4);
percentiles = [50, 75, 90, 95, 99, 99.9];
percentileVals = prctile(absWeights, percentiles);
bar(percentiles, percentileVals, 'FaceColor', [0.7 0.3 0.7]);
xlabel('Percentile', 'FontSize', 11);
ylabel('Weight [Gy]', 'FontSize', 11);
title('Weight Values at Different Percentiles', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;
set(gca, 'YScale', 'log');

% 5. Number of non-zero entries per beamlet
subplot(2, 3, 5);
nonZerosPerBeamlet = full(sum(D ~= 0, 1));
histogram(nonZerosPerBeamlet, 50, 'EdgeColor', 'none', 'FaceColor', [0.9 0.6 0.2]);
xlabel('Non-Zero Voxels per Beamlet', 'FontSize', 11);
ylabel('Number of Beamlets', 'FontSize', 11);
title('Beamlet Influence Spread', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% Add statistics text
meanVoxels = mean(nonZerosPerBeamlet);
medianVoxels = median(nonZerosPerBeamlet);
text(0.95, 0.95, sprintf('Mean: %.0f\nMedian: %.0f', meanVoxels, medianVoxels), ...
    'Units', 'normalized', 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', 'FontSize', 9, 'BackgroundColor', 'white');

% 6. Max weight per beamlet distribution
subplot(2, 3, 6);
maxWeightPerBeamlet = full(max(abs(D), [], 1));
histogram(maxWeightPerBeamlet, 50, 'EdgeColor', 'none', 'FaceColor', [0.4 0.6 0.9]);
xlabel('Max Weight per Beamlet [Gy]', 'FontSize', 11);
ylabel('Number of Beamlets', 'FontSize', 11);
title('Distribution of Maximum Beamlet Weights', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% Add statistics text
meanMaxWeight = mean(maxWeightPerBeamlet);
medianMaxWeight = median(maxWeightPerBeamlet);
text(0.95, 0.95, sprintf('Mean: %.3f Gy\nMedian: %.3f Gy', meanMaxWeight, medianMaxWeight), ...
    'Units', 'normalized', 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', 'FontSize', 9, 'BackgroundColor', 'white');

% Add overall title
sgtitle(sprintf('%s\n%d beamlets x %d voxels, %d non-zero entries (%.2f%% sparse)', ...
    figureTitle, size(D, 2), size(D, 1), numNonZero, ...
    100*(1 - numNonZero/(size(D,1)*size(D,2)))), ...
    'FontSize', 14, 'FontWeight', 'bold');

matRad_cfg.dispInfo('=== Weight distribution analysis complete ===\n\n');

end