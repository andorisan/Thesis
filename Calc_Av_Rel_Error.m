function relative_avg_error = Calc_Av_Rel_Error(tensor_before, tensor_nan, setting, tucker_settings, error_setting)
    
    % Tucker model_________________________________________________________
    if setting == 'tucker'
        [Factors, G, ExplX, Reconstr] = tucker(tensor_nan, [tucker_settings]);
    end %__________________________________________________________________ 
    
    
    
    % Low Multilinear Rank Approximation model_____________________________
    if setting == 'lmlra_'
        [U,G] = lmlra(tensor_nan, [tucker_settings]);

        I=length(tensor_before(:,1,1)); J=length(tensor_before(1,:,1)); K=length(tensor_before(1,1,:));
        Reconstr_temp = zeros(I, J, K);
        Reconstr = zeros(I, J, K);
        
        for t=1:K % K=time frame number (do each time frame individually and add them together afterwards to form a tensor)
            for p=1:length((U{1}(1,:))) %P
                for q=1:length((U{2}(1,:))) %Q
                    for r=1:length((U{3}(1,:))) %R

                            Reconstr_temp(:,:,t) = G(p,q,r)*( (U{1}(:,p)*U{2}(:,q)') * U{3}(t,r));
                            Reconstr(:,:,t) = Reconstr(:,:,t) + Reconstr_temp(:,:,t);

                    end
                end
            end
        end 
    end%___________________________________________________________________
    
    
    
    % Naive model, always guess tensor mean for nan values_________________
    if setting == 'naive_'
        % Calculate average pixel in tensor using available data
        tensor_avg = nansum(nansum(nansum( tensor_nan ))) / sum(sum(sum(~isnan(tensor_nan))));
        % Populate tensor with only the average value
        I=length(tensor_before(:,1,1)); J=length(tensor_before(1,:,1)); K=length(tensor_before(1,1,:));
        Reconstr = zeros(I, J, K);
        for i = 1:I
            for j = 1:J
                for k = 1:K
                   Reconstr(i,j,k) = tensor_avg;
                end
            end
        end   
    end%___________________________________________________________________
    
    
    
    % Model where I start imputing the mean (like the naive model), but____
    % then run Tucker afterwards (Imputation)
    if setting == 'imptuc'
        % Calculate average pixel in tensor using available data
        tensor_avg = nansum(nansum(nansum( tensor_nan ))) / sum(sum(sum(~isnan(tensor_nan))));
        % Impute the average of the tensor where data is missing
        tensor_filled = tensor_nan;
        tensor_filled(isnan(tensor_filled))=tensor_avg;
        [Factors, G, ExplX, Reconstr] = tucker(tensor_filled, [tucker_settings]); 
    end%___________________________________________________________________ 
    
    
    if error_setting == 'RMSE'
        % Now calculate the average relative error where there is missing data
        %find indices of all elements in tensor_nan where there is nan
        tensor_error = (sqrt((Reconstr-tensor_before).^2)).*isnan(tensor_nan);
        tensor_avg_error = sum(sum(sum(tensor_error))) / sum(sum(sum(isnan(tensor_nan))));
        relative_avg_error = tensor_avg_error / mean2(tensor_before);

        %fprintf('The relative average error for a missing pixel: %.4f.\n', relative_avg_error);
    end
    
    
    
    
    if error_setting == 'CORR'
        
        corr_vector = [];
        
        for t = 1:length(tensor_before(1,1,:))
            temp = corr2(tensor_before(:,:,t), Reconstr(:,:,t));
            corr_vector = [corr_vector; temp];
        end    
        
        relative_avg_error = mean(corr_vector);
        %corr_vector = zeros(length(tensor_before(1,1,:)));
        
        %for t = 1:length(tensor_before(1,1,:))
        %    corr_vector(t) = corr2(tensor_before(:,:,t), Reconstr(:,:,t));  
        %    display(length(corr_vector(t)))
        %    %display(corr_vector(t))
        %end
        %display(corr_vector)
        
        %use this variable name for convenience, this is average
        %correlation
        %relative_avg_error = mean(corr_vector);
        
    end
end