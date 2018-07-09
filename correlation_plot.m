function correlation_plot(average_pixels, average_pixels2, average_pixels3, time_loadings, data_name)
   
    
    % Check time rank
    shape = size(time_loadings);
    
    % Rank 1 systems
    correl = corr(time_loadings, average_pixels');
    correl = round(correl,2);
    
    if shape(2) == 1 
        coeffs = polyfit(time_loadings, average_pixels', 1);
        fittedX = linspace(min(time_loadings), max(average_pixels), 2);
        fittedY = polyval(coeffs, fittedX);

        figure
        hold on
        scatter(time_loadings,average_pixels')
        %plot(fittedX, fittedY, 'k-', 'LineWidth', 1);
        refline
        title(sprintf('%s: Correlation between avgerage NDVI \n values and time loadings. r = %.2f', data_name, correl))
        legend({'TL vs Area Mean'}, 'fontsize',14)
        
        
        xlabel('Time loadings','fontsize',14)
        ylabel('Average NDVI','fontsize',14)
        xlim([min(time_loadings)-0.01  max(time_loadings)+0.01])
        xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); %make ticks bigger!
        %ylim([0 250])
        hold off
    end
    
    % Rank 2 systems 
    if shape(2) == 2

        time_loadings1 = time_loadings(:,1);
        time_loadings2 = time_loadings(:,2);
        
        correl1 = corr(time_loadings1, average_pixels');
        correl2 = corr(time_loadings2, average_pixels2');
        
        coeffs = polyfit(time_loadings1, average_pixels', 1);
        fittedX = linspace(min(time_loadings1), max(average_pixels), 2);
        fittedY = polyval(coeffs, fittedX);

        coeffs2 = polyfit(time_loadings2, average_pixels2', 1);
        fittedX2 = linspace(min(time_loadings2), max(average_pixels2), 2);
        fittedY2 = polyval(coeffs2, fittedX2);
        
        figure
        hold on
        scatter(time_loadings1,average_pixels);
        scatter(time_loadings2,average_pixels2);
        refline
        %plot(fittedX, fittedY, 'k-', 'LineWidth', 1);
        %plot(fittedX2, fittedY2, 'k-', 'LineWidth', 1);        
        %title(sprintf('%s: Correlation between avgerage pixel \n per frame and time loadings ', data_name),'fontsize',16)
        
        title(sprintf('%s: Correlation between avgerage NDVI \n values and time loadings. r_1 = %.2f, r_2 = %.2f', data_name, correl1, correl2))
        legend({'TL_1 vs Area_1 Mean', 'TL_2 vs Area_2 Mean'}, 'fontsize',14)

        xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); %make ticks bigger!
        xlabel('Time loadings')
        ylabel('Average NDVI')
        %xlim([  min(min(time_loadings1), min(time_loadings2))-0.01   max(max(time_loadings1), max(time_loadings2))+0.01]  )
        %ylim([50,500])
        hold off    
    end
    
    
    % Rank 3 systems
    if shape(2) == 3
        time_loadings1 = time_loadings(:,1);
        time_loadings2 = time_loadings(:,2);
        time_loadings3 = time_loadings(:,3);
        
        correl1 = corr(time_loadings1, average_pixels'); 
        correl2 = 0;%corr(time_loadings2, average_pixels2');
        correl3 = corr(time_loadings3, average_pixels3');
        
        coeffs = polyfit(time_loadings1, average_pixels', 1);
        fittedX = linspace(min(time_loadings1), max(average_pixels), 2);
        fittedY = polyval(coeffs, fittedX);

        coeffs2 = polyfit(time_loadings2, average_pixels2', 1);
        fittedX2 = linspace(min(time_loadings2), max(average_pixels2), 2);
        fittedY2 = polyval(coeffs2, fittedX2);
        
        coeffs3 = polyfit(time_loadings3, average_pixels3', 1);
        fittedX3 = linspace(min(time_loadings3), max(average_pixels3), 2);
        fittedY3 = polyval(coeffs3, fittedX3);
        
        figure
        hold on
        scatter(time_loadings1,average_pixels)
        scatter(time_loadings2,average_pixels2)
        scatter(time_loadings3,average_pixels3, 'MarkerEdgeColor','g')
        refline

        title(sprintf('%s: Correlation between avgerage NDVI \n values and time loadings. r_1 = %.2f, r_2 = %.2f, r_3 = %.2f', data_name, correl1, correl2, correl3), 'fontsize', 14)
        legend({'TL_1 vs Area_1 Mean', 'TL_2 vs Area_2 Mean', 'TL_3 vs Area_3 Mean'}, 'fontsize',14)
        xlabel('Time loadings', 'fontsize', 14)
        ylabel('Average NDVI', 'fontsize', 14)
        %ylim([50,500])
        %xlim([  min([min(time_loadings1), min(time_loadings2), min(time_loadings3)])-0.01   max([max(time_loadings1), max(time_loadings2), max(time_loadings3])+0.01]  )
        hold off    
    end 
    

end