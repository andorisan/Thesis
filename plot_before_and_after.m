function plot_before_and_after(tensor, reconstruction, not_empty_list, frames_to_plot_list, clims, super_title)
    
    % frames_to_plot_list is chronological, time is the index in our tensors    
    time1 = find(not_empty_list == frames_to_plot_list(1));
    time2 = find(not_empty_list == frames_to_plot_list(2));
    time3 = find(not_empty_list == frames_to_plot_list(3));
    
    figure
    subplot(2,3,1) 
    imagesc(tensor(:,:,time1),clims); colormap(bluewhitered);
    daspect([1 1 1])
    title(['Before, t = ' num2str(frames_to_plot_list(1))])

    subplot(2,3,2) 
    imagesc(tensor(:,:,time2),clims); colormap(bluewhitered);
    daspect([1 1 1])
    title(['Before, t = ' num2str(frames_to_plot_list(2))])
    
    subplot(2,3,3) 
    imagesc(tensor(:,:,time3),clims); colormap(bluewhitered);
    daspect([1 1 1])
    title(['Before, t = ' num2str(frames_to_plot_list(3))])
    
    subplot(2,3,4)
    imagesc(reconstruction(:,:,time1),clims); colormap(bluewhitered);
    daspect([1 1 1])
    title(['Tucker Reconstruction, t =' num2str(frames_to_plot_list(1))])
    
    subplot(2,3,5) 
    imagesc(reconstruction(:,:,time2),clims); colormap(bluewhitered);
    daspect([1 1 1])
    title(['Tucker Reconstruction, t =' num2str(frames_to_plot_list(2))])
    
    subplot(2,3,6) 
    imagesc(reconstruction(:,:,time3),clims); colormap(bluewhitered);
    daspect([1 1 1])
    title(['Tucker Reconstruction, t =' num2str(frames_to_plot_list(3))])
    suptitle(super_title);
    
    % colorbar 
    h = colorbar
    set(h,'Position',[0.95 0.1 0.01 0.76])
end
