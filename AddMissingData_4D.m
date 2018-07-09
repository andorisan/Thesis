function tensor_nan = AddMissingData_4D(tensor, perc_missing_to_add, nan_size, option)

% tensor                = tensor to add missing data to
% perc_missing_to_add   = perc of missing data in output tensor 
% nan_size              = increase the size of missing data pixels by nan_size in each
%                          direction. Use nan_size=0 for single pixel nan patches.
% option                = 'random' to add missing data in different locations, spatially for the different channels
%                         'equal_' to add missing data in the same locations in the different channels (like it is in the
%                         Iberian dataset)

% If we don't want to add any missing data we just break out of the
% function here and return tensor_nan = tensor

% Fix random seed
rng(7);

if perc_missing_to_add == 0
    tensor_nan = tensor;
    perc_nan = 0;
else
    tensor_nan = tensor;


    Y = length(tensor_nan(:,1,1,1));
    X = length(tensor_nan(1,:,1,1));
    T = length(tensor_nan(1,1,:,1));
    C = length(tensor_nan(1,1,1,:));

    perc_nan = 0; i = 0;
    while perc_nan <  perc_missing_to_add
        Yrand=randi(Y); Xrand=randi(X); Trand=randi(T); 
        
        ymin = max(1, Yrand-nan_size); ymax=min(Y, Yrand+nan_size);   
        xmin = max(1, Xrand-nan_size); xmax=min(X, Xrand+nan_size); 
        
        if option == 'random'
            Crand=randi(C);
            tensor_nan(ymin:ymax, xmin:xmax, Trand, Crand) = nan;
        end
        
        if option == 'equal_'
            tensor_nan(ymin:ymax, xmin:xmax, Trand, 1) = nan;
            tensor_nan(ymin:ymax, xmin:xmax, Trand, 2) = nan;
        end

        % Calculate percentage nan
        perc_nan = sum(sum(sum(sum(isnan(tensor_nan(:,:,:,:)))))) / numel(tensor_nan);
        i=i+1;
%         if rem(i,200) == 0   %uncomment to print progress when working with large tensors
%             disp(perc_nan);
%         end
    end

    %display(perc_nan)
end


% %  Show before and after adding missing data
%    minVal = nanmin(nanmin(nanmin( tensor_nan(:,:,1,1)  ))) -0.2; %-.2 to make plot look more smooth
%    maxVal = nanmax(nanmax(nanmax( tensor_nan(:,:,1,1)  )));
%    clims=[minVal, maxVal];
% % 
% %  figure
%   subplot(1,2,1)
%   imagesc(tensor(:,:,1,1),clims), daspect([1 1 1]), title('Before adding missing data')
%   subplot(1,2,2)
%   imagesc(tensor_nan(:,:,1,1),clims),daspect([1 1 1]), title(['Perc. missing data =' num2str(perc_nan)]);
%   colormap hot;
% 
% % figure
%  imagesc(tensor_nan(:,:,1,1),clims),daspect([1 1 1]), title(['Ratio missing data =' num2str(perc_nan)]);


end 