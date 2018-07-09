function VideoPlot(tensor_before, tensor_reconstr, tensor_factors, pause_length)
    
    clims1 = [-0.35, 0.35]
    clims2 = [-0.35, 0.35]%[nanmin(nanmin(nanmin(tensor_before(:,:,1) - tensor_reconstr(:,:,1)))), nanmax(nanmax(nanmax(tensor_before(:,:,1) - tensor_reconstr(:,:,1))))]
    %set(0,'DefaultFigureColormap',feval('bluewhitered')); 
    nframes=length(tensor_reconstr(1,1,:)) 
    %mov(1:nframes)= struct('cdata',[],'colormap',[]);
    set(gca,'nextplot','replacechildren')
    
    for i=1:nframes
      pause(pause_length);
      
      % Plot original tensor
      subplot(2,3,1)
      imagesc(tensor_before(:,:,i), clims1); daspect([1 1 1]); colormap(bluewhitered);
      title('Original Tensor')
      mov(i)=getframe(gcf);
      
      % Plot the reconstruction
      subplot(2,3,2)
      imagesc(tensor_reconstr(:,:,i), clims1); daspect([1 1 1]);
      title('Reconstruction')
      
      % Plot Residuals
      subplot(2,3,3)
      imagesc(abs((tensor_reconstr(:,:,i) -  tensor_before(:,:,i))), clims2); daspect([1 1 1]);
      title('Abs. Residuals')
      
      ax2 = subplot(2,3,[4 5 6]) 
      axis manual
      plot(tensor_factors{3})
      ylim([-0.2,0.4])
      title('Time Loadings')
      legend({'TL1: Shrubs', 'TL2: Plastic'},'Location','northwest','AutoUpdate','off')
      axes(ax2)
      line([i i],get(ax2,'YLim'),'Color',[0 0 0]);
      mov(i)=getframe(gcf);
    end

    %VideoWriter(mov, '1moviename.avi', 'compression', 'None');

end 