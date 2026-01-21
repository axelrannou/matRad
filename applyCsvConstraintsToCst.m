function cst = applyCsvConstraintsToCst(csvFilePath, cst, numFractions)
    % Read the CSV
    constraints = readtable(csvFilePath, 'Delimiter', ',');
    
    % Apply constraints
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
        
        % Process PTV constraints - handle each line separately
        if isPTV && strcmpi(type, 'Point')
            entry = struct();
            entry.penalty = priority;
            
            if strcmpi(limit, 'inférieur') && volume == 100.0
                % This is a mean dose target (100% at dose X)
                entry.className = 'DoseObjectives.matRad_SquaredDeviation';
                entry.parameters = {dosePerFraction};
            elseif strcmpi(limit, 'inférieur') && volume < 100.0
                % This is a DVH constraint (e.g., D98% >= X)
                entry.className = 'DoseObjectives.matRad_MinDVH';
                entry.parameters = {dosePerFraction, volume};
            elseif strcmpi(limit, 'supérieur')
                % This is a max dose constraint
                entry.className = 'DoseObjectives.matRad_MaxDVH';
                entry.parameters = {dosePerFraction, volume};
            else
                warning('Unknown PTV constraint: %s at %.1f%%', limit, volume);
                continue;
            end
            
            % Append to existing or create new entry
            if isempty(cst{cstIndex,6})
                cst{cstIndex,6} = {entry};
            else
                cst{cstIndex,6}{end+1} = entry;
            end
            
            % Skip to next constraint
            continue;
        end
        
        % For non-PTVs or gEUD constraints
        entry = struct(); 
        entry.penalty = priority;
        
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
