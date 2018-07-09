function relative_avg_error = Calc_Av_Rel_Error_Ranks(tensor, iter, setting)
    
    
    if setting == 'spatial_'
        [Factors, G, ExplX, Reconstr] = tucker(tensor, [iter, iter, 2]);
    end
    
    if setting == 'temporal'
        [Factors, G, ExplX, Reconstr] = tucker(tensor, [30,30, iter]);
    end
    
    % Now calculate the average relative error where there is missing data
    %find indices of all elements in tensor_nan where there is nan
    tensor_error = (sqrt((Reconstr-tensor).^2));
    tensor_avg_error = sum(sum(sum(tensor_error))) / numel(tensor);
    relative_avg_error = tensor_avg_error / mean2(tensor);
   

end