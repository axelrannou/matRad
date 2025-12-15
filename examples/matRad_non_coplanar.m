%% Example: Reading and displaying CSV data
% This script demonstrates how to read and display CSV data in MATLAB
% using the specified constraints data file

% Define the file path to the CSV file
%csvFilePath = '/home/arannou/Documents/Workspace/matrad_env/matRad/userdata/patients/constraints_data_jane_doe.csv';
%csvFilePath = '/home/arannou/Documents/Workspace/matrad_env/matRad/userdata/patients/constraints_data_john_doe.csv';
csvFilePath = '/home/arannou/Documents/Workspace/matrad_env/matRad/userdata/patients/constraints_data_john_doe_lung.csv';

% Check if the file exists
if ~exist(csvFilePath, 'file')
    error('CSV file not found at: %s', csvFilePath);
end

% Read the CSV file
try
    % Method 1: Using readtable (recommended for newer MATLAB versions)
    dataTable = readtable(csvFilePath);

    % Display the table
    disp('CSV data as table:');
    disp(dataTable);

    % Method 3: Using textscan for more control
    fileID = fopen(csvFilePath, 'r');
    if fileID == -1
        error('Could not open the file for reading');
    end

    % Read the first line to get headers
    headerLine = fgetl(fileID);
    if ~ischar(headerLine)
        error('Could not read header line from the file');
    end

    headers = strsplit(headerLine, ',');
    disp('CSV Headers:');
    disp(headers);

    % Reset file position and read all data as cell array
    frewind(fileID);
    fileContent = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);

    % Process and display the content
    disp('CSV content line by line:');
    for i = 1:length(fileContent{1})
        disp(fileContent{1}{i});
    end

catch readErr
    error('Failed to read the CSV file: %s', readErr.message);
end

% Additional processing can be done here as needed

%% Patient Data Import
%load('jane_doe.mat');
%load('john_doe.mat');
load('john_doe_lung.mat');

patientName = 'john_doe_lung';  % Change to 'john_doe' if using that file


disp('cst values:');
% Make cst{i,6} empty
for i = 1:size(cst,1)
    if ~isempty(cst{i,6}) && iscell(cst{i,6})
        cst{i,6} = [];
    end
end

% For each entry in cst, safely print cst{i,6} values
for i = 1:size(cst,1)
    try
        % Get organ name from cst{i,2}
        if ~isempty(cst{i,2})
            organName = cst{i,2};
        else
            organName = 'Unnamed organ';
        end

        % Display organ name and constraint value
        if ~isempty(cst{i,6}) && iscell(cst{i,6})
            constraintValue = cst{i,6}{1};
            % Handle different data types for the constraint value
            if isstruct(constraintValue)
                disp([organName ': <struct value>']);
                disp(['  Fields: ' strjoin(fieldnames(constraintValue), ', ')]);

                % Display the values inside each field of the struct
                structFields = fieldnames(constraintValue);
                for j = 1:length(structFields)
                    fieldName = structFields{j};
                    fieldValue = constraintValue.(fieldName);

                    if isnumeric(fieldValue)
                        disp(['    ' fieldName ': ' num2str(fieldValue)]);
                    elseif ischar(fieldValue)
                        disp(['    ' fieldName ': ' fieldValue]);
                    elseif iscell(fieldValue)
                        disp(['    ' fieldName ': <cell array>']);
                        % Display cell array contents
                        for cellIdx = 1:numel(fieldValue)
                            cellValue = fieldValue{cellIdx};
                            if isnumeric(cellValue)
                                disp(['      Element ' num2str(cellIdx) ': ' num2str(cellValue)]);
                            elseif ischar(cellValue)
                                disp(['      Element ' num2str(cellIdx) ': ' cellValue]);
                            elseif isstruct(cellValue)
                                disp(['      Element ' num2str(cellIdx) ': <struct>']);
                                % Display the struct inside the cell
                                nestedFields = fieldnames(cellValue);
                                for k = 1:length(nestedFields)
                                    nestedName = nestedFields{k};
                                    nestedValue = cellValue.(nestedName);
                                    if isnumeric(nestedValue)
                                        disp(['        ' nestedName ': ' num2str(nestedValue)]);
                                    elseif ischar(nestedValue)
                                        disp(['        ' nestedName ': ' nestedValue]);
                                    else
                                        disp(['        ' nestedName ': <' class(nestedValue) '>']);
                                    end
                                end
                            else
                                disp(['      Element ' num2str(cellIdx) ': <' class(cellValue) '>']);
                            end
                        end
                    elseif isstruct(fieldValue)
                        disp(['    ' fieldName ': <nested struct>']);
                        % Display nested struct fields
                        nestedFields = fieldnames(fieldValue);
                        for k = 1:length(nestedFields)
                            nestedName = nestedFields{k};
                            nestedValue = fieldValue.(nestedName);
                            if isnumeric(nestedValue)
                                disp(['      ' nestedName ': ' num2str(nestedValue)]);
                            elseif ischar(nestedValue)
                                disp(['      ' nestedName ': ' nestedValue]);
                            else
                                disp(['      ' nestedName ': <' class(nestedValue) '>']);
                            end
                        end
                    else
                        disp(['    ' fieldName ': <' class(fieldValue) '>']);
                    end
                end
            elseif isnumeric(constraintValue)
                disp([organName ': ' num2str(constraintValue)]);
            elseif ischar(constraintValue)
                disp([organName ': ' constraintValue]);
            else
                disp([organName ': <value of type ' class(constraintValue) '>']);
            end
        else
            disp([organName ': <empty or non-cell value>']);
        end
    catch err
        disp(['Organ at index ' num2str(i) ': <error displaying value - ' err.message '>']);
    end
end


%% Treatment Plan
pln.radiationMode   = 'photons';
pln.machine         = 'Generic';
pln.numOfFractions  = 30;

% Define gantry angles with 4-degree spacing
gantryAngles = [0:4:359];

% Define three couch angles for arc planning
couchAngles = [-5, 0, 5];

% Create all combinations of gantry and couch angles
pln.propStf.gantryAngles = [];
pln.propStf.couchAngles = [];

for couchAngle = couchAngles
    pln.propStf.gantryAngles = [pln.propStf.gantryAngles, gantryAngles];
    pln.propStf.couchAngles = [pln.propStf.couchAngles, repmat(couchAngle, 1, numel(gantryAngles))];
end

pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles);
pln.propStf.bixelWidth = 7;
pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);

pln.bioModel = 'none'; 
pln.multScen = 'nomScen';
pln.propStf.continuousAperture = false;

%% Dose grid
pln.propDoseCalc.doseGrid.resolution.x = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.y = 3; % [mm]
pln.propDoseCalc.doseGrid.resolution.z = 3; % [mm]

% Optimizer settings
pln.propOpt.optimizer = 'PGD_CPU';
pln.propOpt.quantityOpt = 'physicalDose';
pln.propSeq.runSequencing = false;
pln.propOpt.runDAO = false;

%% Apply Constraints
cst = applyCsvConstraintsToCst(csvFilePath, cst, pln.numOfFractions);

%% Generate beam geometry
stf = matRad_generateStf(ct,cst,pln);

%% Dose Calculation
dijFolder = fullfile(pwd, 'dij_cache');
if ~exist(dijFolder, 'dir')
    mkdir(dijFolder);
end

% Use patient name, number of beams, and resolution in the identifier
dijFile = fullfile(dijFolder, sprintf('%s_dij_%dbeams_res%dmm.mat', ...
    patientName, pln.propStf.numOfBeams, pln.propDoseCalc.doseGrid.resolution.x));

if exist(dijFile, 'file')
    disp(['Loading precomputed dij from ' dijFile]);
    load(dijFile, 'dij');
else
    disp('Calculating dij');
    disp('This will take a while but only needs to be done once...');
    tic;
    dij = matRad_calcDoseInfluence(ct,cst,stf,pln);
    calcTime = toc;
    disp(['Calculation took ' num2str(calcTime/60) ' minutes']);
    save(dijFile, 'dij', '-v7.3');
    disp(['Saved dij to ' dijFile]);
end

%%Final Inverse Planning
resultGUI = matRad_fluenceOptimization(dij,cst,pln);


%% Display dose slices with CT overlay using matRad plotting functions
fprintf('Creating enhanced dose slice visualizations using matRad functions...\n');

% Create a new figure for dose slices
figure('Name', 'Dose Distribution with CT', 'Position', [50, 50, 1200, 1000]);

% Get dose dimensions
[nx, ny, nz] = size(resultGUI.physicalDose);

% Calculate middle slices - only need axial
slices = struct('axial', round(nz-100));
plane = 3; % axial only
slice = slices.axial;

% Get colormap settings
doseColorMap = jet(64);
ctColorMap = bone(64);

% Define dose display parameters
doseThreshold = 0.1; % 10% threshold
doseAlpha = 0.6;
doseWindow = [0 max(resultGUI.physicalDose(:))];

% Define which structures to display (all visible ones)
voiSelection = true(size(cst,1), 1);
for i = 1:size(cst, 1)
    if strcmp(cst{i,3}, 'IGNORED')
        voiSelection(i) = false;
    end
end

% Plot axial view only
ax = gca;

% Use matRad's wrapper function to plot everything
try
    [~, ~, ~, ~, ~] = matRad_plotSliceWrapper(ax, ct, cst, 1, resultGUI.physicalDose, ...
        plane, slice, doseThreshold, doseAlpha, [], doseColorMap, doseWindow, [], ...
        voiSelection, 'Dose [Gy]', false);
    
    title(sprintf('Axial View (Slice %d/%d)', slice, size(resultGUI.physicalDose, plane)), ...
        'FontSize', 14, 'FontWeight', 'bold');
catch ME
    % Fallback to simple visualization if matRad functions fail
    warning('Could not use matRad plotting functions: %s', ME.message);
    
    doseSlice = resultGUI.physicalDose(:, :, slice)';
    
    imagesc(doseSlice);
    colormap(ax, jet);
    colorbar;
    title(sprintf('Axial View (Slice %d/%d)', slice, size(resultGUI.physicalDose, plane)));
    axis equal tight;
end

fprintf('Dose slice visualization complete!\n');

%% Visualization and Analysis
%matRadGUI;

%% Indicator Calculation and display of DVH and QI
resultGUI = matRad_planAnalysis(resultGUI,ct,cst,stf,pln);

% Save final results
%save('final_optimized_plan.mat', 'resultGUI', 'pln', 'stf', 'dij', 'cst', 'ct', ...
%    'selectedGantryAngles', 'selectedCouchAngles', '-v7.3');
%disp('Final plan saved to final_optimized_plan.mat');
