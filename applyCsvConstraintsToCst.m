function cst = applyCsvConstraintsToCst(csvFilePath, cst, numFractions)
    % Read the CSV
    constraints = readtable(csvFilePath, 'Delimiter', ',');
    
    % Create a mapping to store min/max dose for PTVs
    ptvDoseMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    
    % First pass to collect PTV min/max values
    for i = 1:height(constraints)
        structureName = strtrim(constraints.Structure{i});
        type = strtrim(constraints.Type{i});
        limit = strtrim(constraints.Limite{i});
        doseTotal = constraints.Gy(i);
        
        % Only interested in PTVs with Point type in this first pass
        isPTV = contains(lower(structureName), 'ptv');
        if isPTV && strcmpi(type, 'Point')
            % Initialize if needed
            if ~ptvDoseMap.isKey(structureName)
                ptvDoseMap(structureName) = [Inf, 0]; % [min, max]
            end
            
            % Calculate dose per fraction correctly
            dosePerFraction = doseTotal * numFractions;
            
            % Update min/max values
            currentValues = ptvDoseMap(structureName);
            if strcmpi(limit, 'inférieur')
                currentValues(1) = dosePerFraction; % min dose
            elseif strcmpi(limit, 'supérieur')
                currentValues(2) = dosePerFraction; % max dose
            end
            ptvDoseMap(structureName) = currentValues;
        end
    end
    
    % Second pass to apply constraints
    for i = 1:height(constraints)
        structureName = strtrim(constraints.Structure{i});
        type = strtrim(constraints.Type{i});
        limit = strtrim(constraints.Limite{i});
        doseTotal = constraints.Gy(i);
        volume = constraints.Vol(i);
        priority = constraints.Priorite(i);
        penalty = 1; % Use priority as penalty
        gEUD = constraints.gEUDa(i);
        
        % Convert dose correctly
        dosePerFraction = doseTotal * numFractions;
        
        % Determine if structure is a PTV
        isPTV = contains(lower(structureName), 'ptv');
        
        % Find matching index in cst
        cstIndex = find(strcmpi(cst(:,2), structureName), 1);
        
        if isempty(cstIndex)
            warning('Structure %s not found in cst. Skipping...', structureName);
            continue;
        end
        
        % Skip PTV individual constraints, we'll add them once per PTV
        if isPTV && strcmpi(type, 'Point') && ptvDoseMap.isKey(structureName)
            % Only process each PTV once
            if strcmpi(limit, 'inférieur')
                % Build MinMaxEUD constraint for PTV
                entry = struct();
                entry.epsilon = 0.001; % Default epsilon
                entry.className = 'DoseConstraints.matRad_MinMaxDose';
                
                doseValues = ptvDoseMap(structureName);
                doseMin = doseValues(1);
                doseMax = doseValues(2);
                
                entry.parameters = {doseMin, doseMax, 1}; 
                
                % Add to cst
                if isempty(cst{cstIndex,6})
                    cst{cstIndex,6} = {entry};
                end
                
                % Add objective for PTV optimization
                %objectiveEntry = struct();
                %objectiveEntry.type = 'square deviation';
                %objectiveEntry.dose = (doseMin + doseMax) / 2; % Target dose (middle of min/max)
                %objectiveEntry.weight = 100; % High weight for target
                %objectiveEntry.penalty = 1;
                %objectiveEntry.className = 'DoseObjectives.matRad_SquaredDeviation';
                %objectiveEntry.parameters = {objectiveEntry.dose};
                
                % Add both constraint AND objective
                cst{cstIndex,6} = {entry}; % Both constraint and objective
                
                % Remove this PTV from the map so we don't process it again
                ptvDoseMap.remove(structureName);
            end
            
            % Skip to next constraint
            continue;
        end
        
        % For non-PTVs or gEUD constraints
        entry = struct();
        entry.penalty = penalty;
        
        if strcmpi(type, 'Point') && ~isPTV
            if strcmpi(limit, 'supérieur')
                entry.className = 'DoseObjectives.matRad_MaxDVH';
                entry.parameters = {dosePerFraction, volume}; % Convert % to fraction
            elseif strcmpi(limit, 'inférieur')
                entry.className = 'DoseObjectives.matRad_MinDVH';
                entry.parameters = {dosePerFraction, volume}; % Convert % to fraction
            else
                warning('Unknown limit for Point type: %s', limit);
                continue;
            end
        elseif strcmpi(type, 'gEUD')
            if isnan(gEUD)
                gEUD = 1.0; % Default if not specified
            end
            entry.className = 'DoseObjectives.matRad_EUD';
            entry.parameters = {dosePerFraction, gEUD};
        else
            warning('Unknown type: %s', type);
            continue;
        end
        
        % Append to existing or create new entry
        if isempty(cst{cstIndex,6})
            cst{cstIndex,6} = {entry};
        else
            cst{cstIndex,6}{end+1} = entry;
        end
    end
    
    disp('CSV constraints successfully applied to cst.');
end
