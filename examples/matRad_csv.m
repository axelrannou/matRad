%% Example: Reading and displaying CSV data
% This script demonstrates how to read and display CSV data in MATLAB
% using the specified constraints data file

% Define the file path to the CSV file
csvFilePath = '/home/arannou/Documents/Workspace/matrad_env/matRad/userdata/patients/constraints_data_jane_doe.csv';
%csvFilePath = '/home/arannou/Documents/Workspace/matrad_env/matRad/userdata/patients/constraints_data_john_doe.csv';

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
load('jane_doe.mat');
%load('john_doe.mat');

%% Treatment Plan
% The next step is to define your treatment plan labeled as 'pln'. This 
% structure requires input from the treatment planner and defines 
% the most important cornerstones of your treatment plan.

pln.radiationMode   = 'photons';   % either photons / protons / carbon
pln.machine         = 'Generic';
pln.numOfFractions  = 30;
 
pln.propStf.gantryAngles    = [0:20:359];
pln.propStf.couchAngles     = zeros(1,numel(pln.propStf.gantryAngles));
pln.propStf.bixelWidth      = 7;
pln.propStf.numOfBeams      = numel(pln.propStf.gantryAngles);
pln.propStf.isoCenter       = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct,0);

pln.bioModel = 'none'; 
pln.multScen = 'nomScen';
pln.propStf.continuousAperture  = false;

% dose calculation settings
pln.propDoseCalc.doseGrid.resolution.x = 2; % [mm]
pln.propDoseCalc.doseGrid.resolution.y = 2; % [mm]
pln.propDoseCalc.doseGrid.resolution.z = 2; % [mm]

% We can also use other solver for optimization than IPOPT. matRad 
% currently supports fmincon from the MATLAB Optimization Toolbox. First we
% check if the fmincon-Solver is available, and if it es, we set in in the
% pln.propOpt.optimizer vairable. Otherwise wie set to the default
% optimizer 'IPOPT'
pln.propOpt.optimizer = 'PGD_GPU';
%if matRad_OptimizerFmincon.IsAvailable()
%    pln.propOpt.optimizer = 'fmincon';   
%else
%    pln.propOpt.optimizer = 'IPOPT';
%    
%    Set IPOPT specific options to increase iterations
%    pln.propOpt.ipopt.max_iter = 10000;       % Increase from default 500 to 2000
%    pln.propOpt.ipopt.realtimeUpdate = false; % Disable real-time plot updates during optimization
%
%end
pln.propOpt.quantityOpt = 'physicalDose';  

%%
% Enable sequencing and direct aperture optimization (DAO).
pln.propSeq.runSequencing = false;
pln.propOpt.runDAO        = false;

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

%% Objectives and Constraints
%csvFilePath = '/home/arannou/Documents/Workspace/matrad_env/matRad/userdata/patients/constraints_data.csv';
cst = applyCsvConstraintsToCst(csvFilePath, cst, pln.numOfFractions);

%% Generate Beam Geometry STF
stf = matRad_generateStf(ct,cst,pln);

%% Dose Calculation
% Lets generate dosimetric information by pre-computing dose influence 
% matrices for unit beamlet intensities. Having dose influences available 
% allows for subsequent inverse optimization.
% --- Dose Influence Matrix Caching ---
dijFolder = fullfile(pwd, 'dij_cache');
if ~exist(dijFolder, 'dir')
    mkdir(dijFolder);
end

angleStep = pln.propStf.gantryAngles(2) - pln.propStf.gantryAngles(1);
dijFile = fullfile(dijFolder, sprintf('dij_%d.mat', angleStep));

if exist(dijFile, 'file')
    disp(['Loading precomputed dij from ' dijFile]);
    load(dijFile, 'dij');
else
    disp('Calculating dij...');
    dij = matRad_calcDoseInfluence(ct,cst,stf,pln);
    save(dijFile, 'dij', '-v7.3');  % Add -v7.3 flag for large variables
    disp(['Saved dij to ' dijFile]);
end

maxNumCompThreads(8);
%% Inverse Planning
% The goal of the fluence optimization is to find a set of beamlet weights 
% which yield the best possible dose distribution according to the 
% predefined clinical objectives and constraints underlying the radiation 
% treatment. Once the optimization has finished, trigger once the GUI to
% visualize the optimized dose cubes.
resultGUI = matRad_fluenceOptimization(dij,cst,pln);

%% Sequencing
% This is a multileaf collimator leaf sequencing algorithm that is used in 
% order to modulate the intensity of the beams with multiple static 
% segments, so that translates each intensity map into a set of deliverable 
% aperture shapes.
%resultGUI = matRad_sequencing(resultGUI,stf,dij,pln);

%% DAO - Direct Aperture Optimization
% The Direct Aperture Optimization is an optimization approach where we 
% directly optimize aperture shapes and weights.
%resultGUI = matRad_directApertureOptimization(dij,cst,resultGUI.apertureInfo,resultGUI,pln);
matRadGUI;

%% Aperture visualization
% Use a matrad function to visualize the resulting aperture shapes
%matRad_visApertureInfo(resultGUI.apertureInfo);

%% Indicator Calculation and display of DVH and QI
resultGUI = matRad_planAnalysis(resultGUI,ct,cst,stf,pln);
