function plot_loadings_chronologically(tensor_factors, tensor_factors_both, notEmpty, color, plot_title)
    
    %tensor_factors_both = factors for rank 2 systems. Only used in the
    %rank 1 example to determine ylim for plots
    %mini = min(min(min(tensor_factors{3}), min(min(tensor_factors_both{3}))));
    %maxi = max(max(max(tensor_factors{3}), max(max(tensor_factors_both{3}))));
    
    % Check time rank
    %shape = size(tensor_factors{3});
    shape = 1;
    
    %________________________________
    if shape ==1
    %if shape(2) == 1    
        % Populate dictionary
        factor_dict = containers.Map
        j = 1;
        for i = 1:366
            key_index = num2str(i);

            if (ismember(i,notEmpty) == 0)
                factor_dict(key_index) = NaN;
            else
                factor_dict(key_index) = tensor_factors(j); %tensor_factors{3}(j);
                j = j+1;
            end 
        end 

        % Extract values and keys from dictionary into arrays
        factor_array_vals = [];
        factor_array_keys = [];

        for i = 1:366
            key_index = num2str(i);
            factor_array_vals = [factor_array_vals, factor_dict(key_index)];
            factor_array_keys = [factor_array_keys, i];
        end

        I = ~isnan(factor_array_keys) & ~isnan(factor_array_vals);

        %figure
        display('as?dlfkasdf')
        plot(factor_array_keys(I),factor_array_vals(I), color); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
        title(plot_title,'fontsize',16);
        xticks([1 32 60 91 121 152 182 213 243 274 305 336 ]);
        xticklabels({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul' 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'});
        xtickangle(35)
        %legend({'Time loading 1'}, 'fontsize',14)
        %ylim([-0.1 0.25]);
        
        % For average NDVI plots
        ylim([-0.05 0.3]);
        ylabel('Average NDVI')
    end 
    
    
    %________________________________
    if shape(2) == 2    
        
        % Scale loading 2 to see it better
        tensor_factors{3}(:,1) = tensor_factors{3}(:,1)*(1.2);
        
        % Populate dictionary
        factor_dict  = containers.Map;
        factor_dict2 = containers.Map;
        
        j = 1;
        for i = 1:366
            key_index = num2str(i);

            if (ismember(i,notEmpty) == 0)
                factor_dict(key_index)  = NaN;
                factor_dict2(key_index) = NaN;
            else
                factor_dict(key_index) = tensor_factors{3}(j,1);
                factor_dict2(key_index) = tensor_factors{3}(j,2);
                j = j+1;
            end 
        end 

        % Extract values and keys from dictionary into arrays
        factor_array_vals = [];
        factor_array_vals2= [];
        factor_array_keys = [];
        factor_array_keys2= [];

        for i = 1:366
            key_index = num2str(i);
            factor_array_vals  = [factor_array_vals,  factor_dict(key_index)];
            factor_array_vals2 = [factor_array_vals2, factor_dict2(key_index)];
            factor_array_keys  = [factor_array_keys,  i];
            factor_array_keys2 = [factor_array_keys2, i];
        end

        I = ~isnan(factor_array_keys)   & ~isnan(factor_array_vals);
        I2 = ~isnan(factor_array_keys2) & ~isnan(factor_array_vals2);
        
        %figure
        plot(factor_array_keys(I),factor_array_vals(I), 'g', factor_array_keys2(I2),factor_array_vals2(I2), 'r');  xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
        title(plot_title,'fontsize',14);
        xticks([1 32 60 91 121 152 182 213 243 274 305 336 ]);
        xticklabels({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul' 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'});
        xtickangle(35)
        ylim( [mini, maxi]);
        xlim([1 366])
        
        legend({'TL1: Shrubs', 'TL2: Plastic'}, 'fontsize', 14,'AutoUpdate','off')
    end     
    
    %________________________________
    if shape(2) == 3 
        
        % Scale loading 1 to see it better
        tensor_factors{3}(:,1) = tensor_factors{3}(:,1)*(1.1);
        
        % Scale loading 3 to see it better
        tensor_factors{3}(:,3) = tensor_factors{3}(:,3)*(0.8);
        
        % Populate dictionary
        factor_dict  = containers.Map;
        factor_dict2 = containers.Map;
        factor_dict3 = containers.Map;
        
        j = 1;
        for i = 1:366
            key_index = num2str(i);

            if (ismember(i,notEmpty) == 0)
                factor_dict(key_index)  = NaN;
                factor_dict2(key_index) = NaN;
                factor_dict3(key_index) = NaN;
            else
                factor_dict(key_index) = tensor_factors{3}(j,1);
                factor_dict2(key_index) = tensor_factors{3}(j,2);
                factor_dict3(key_index) = tensor_factors{3}(j,3);
                j = j+1;
            end 
        end 

        % Extract values and keys from dictionary into arrays
        factor_array_vals = [];
        factor_array_vals2= [];
        factor_array_vals3= [];
        
        factor_array_keys = [];
        factor_array_keys2= [];
        factor_array_keys3= [];

        for i = 1:366
            key_index = num2str(i);
            factor_array_vals  = [factor_array_vals,  factor_dict(key_index)];
            factor_array_vals2 = [factor_array_vals2, factor_dict2(key_index)];
            factor_array_vals3 = [factor_array_vals3, factor_dict3(key_index)];
            factor_array_keys  = [factor_array_keys,  i];
            factor_array_keys2 = [factor_array_keys2, i];
            factor_array_keys3 = [factor_array_keys3, i];
        end

        I = ~isnan(factor_array_keys)   & ~isnan(factor_array_vals);
        I2 = ~isnan(factor_array_keys2) & ~isnan(factor_array_vals2);
        I3 = ~isnan(factor_array_keys3) & ~isnan(factor_array_vals3);
        
        %figure
        plot(factor_array_keys(I),factor_array_vals(I), 'g', factor_array_keys2(I2),factor_array_vals2(I2), 'r',...
            factor_array_keys3(I3),factor_array_vals3(I3), 'blue');
        title(plot_title,'fontsize',16);
        xticks([1 32 60 91 121 152 182 213 243 274 305 336 ]);
        xticklabels({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul' 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'});
        xtickangle(35)
        xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
        ylim( [mini, maxi]);
        legend({'Time loading 1', 'Time loading 2', 'Time loading 3'}, 'fontsize', 14)
    end

    
    %________________________________
    if shape(2) == 4    
        % Populate dictionary
        factor_dict  = containers.Map; factor_dict2 = containers.Map;
        factor_dict3 = containers.Map; factor_dict4 = containers.Map;
        
        j = 1;
        for i = 1:366
            key_index = num2str(i);

            if (ismember(i,notEmpty) == 0)
                factor_dict(key_index)  = NaN; factor_dict2(key_index) = NaN;
                factor_dict3(key_index) = NaN; factor_dict4(key_index) = NaN;
            else
                factor_dict(key_index) = tensor_factors{3}(j,1);
                factor_dict2(key_index) = tensor_factors{3}(j,2);
                factor_dict3(key_index) = tensor_factors{3}(j,3);
                factor_dict4(key_index) = tensor_factors{3}(j,4);
                j = j+1;
            end 
        end 

        % Extract values and keys from dictionary into arrays
        factor_array_vals = [];
        factor_array_vals2= [];
        factor_array_vals3= [];
        factor_array_vals4= [];
        
        factor_array_keys = [];
        factor_array_keys2= [];
        factor_array_keys3= [];
        factor_array_keys4= [];

        for i = 1:366
            key_index = num2str(i);
            factor_array_vals  = [factor_array_vals,  factor_dict(key_index)];
            factor_array_vals2 = [factor_array_vals2, factor_dict2(key_index)];
            factor_array_vals3 = [factor_array_vals3, factor_dict3(key_index)];
            factor_array_vals4 = [factor_array_vals4, factor_dict4(key_index)];
            factor_array_keys  = [factor_array_keys,  i];
            factor_array_keys2 = [factor_array_keys2, i];
            factor_array_keys3 = [factor_array_keys3, i];
            factor_array_keys4 = [factor_array_keys4, i];
        end

        I = ~isnan(factor_array_keys)   & ~isnan(factor_array_vals);
        I2 = ~isnan(factor_array_keys2) & ~isnan(factor_array_vals2);
        I3 = ~isnan(factor_array_keys3) & ~isnan(factor_array_vals3);
        I4 = ~isnan(factor_array_keys4) & ~isnan(factor_array_vals4);
        
        %figure
        plot(factor_array_keys(I),factor_array_vals(I), 'g', factor_array_keys2(I2),factor_array_vals2(I2), 'r',... 
            factor_array_keys3(I3),factor_array_vals3(I3), 'blue', factor_array_keys4(I4),factor_array_vals4(I4), 'y');
        xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
        title(plot_title,'fontsize',16);
        xticks([1 32 60 91 121 152 182 213 243 274 305 336 ]);
        xticklabels({'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul' 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'});
        xtickangle(35)
        ylim( [mini, maxi]);
        legend({'Time loading 1', 'Time loading 2', 'Time loading 3', 'Time loading 4'}, 'fontsize', 14)
    end    
    
end 