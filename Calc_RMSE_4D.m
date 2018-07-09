function rmse = Calc_RMSE_4D(tensor_nan, Reconstr, tensor_before)
    
    % Compress the 4D case before calculating error
    % Note: The 3D case was already compressed before running Tucker
    if ndims(Reconstr) == 4
        Reconstr = (Reconstr(:,:,:,2)-Reconstr(:,:,:,1)) ./ (Reconstr(:,:,:,2)+Reconstr(:,:,:,1));
    end
    
    tensor_error = (sqrt((Reconstr-tensor_before).^2)) .*  (isnan(tensor_nan));
    tensor_avg_error = sum(sum(sum(tensor_error))) / sum(sum(sum(isnan(  tensor_nan ))));
    rmse = tensor_avg_error / mean2(tensor_before);

end 