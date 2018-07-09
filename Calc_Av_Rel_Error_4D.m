function relative_avg_error_4D = Calc_Av_Rel_Error(tensor_before, tensor_nan, setting, tucker_settings)
    
    % Tucker model_________________________________________________________
    if setting == 'tucker'
        [Factors, G, ExplX, Reconstr_4D] = tucker(tensor_nan, [tucker_settings]);
    end %__________________________________________________________________     
    
    % Calculate NDVI after running the model
    Reconstr = (Reconstr4D(:,:,:,2)-Reconstr4D(:,:,:,1))./(Reconstr4D(:,:,:,2)+Reconstr4D(:,:,:,1));
    
    
    % Now calculate the average relative error where there is missing data
    %find indices of all elements in tensor_nan where there is nan
    tensor_error = (sqrt((Reconstr-tensor_before).^2)).*isnan(tensor_nan);
    tensor_avg_error = sum(sum(sum(tensor_error))) / sum(sum(sum(isnan(tensor_nan))));
    relative_avg_error = tensor_avg_error / mean2(tensor_before);

    %fprintf('The relative average error for a missing pixel: %.4f.\n', relative_avg_error);

end