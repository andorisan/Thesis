%% Chapter 6: Results

%% Read data
clc; clear all; close all;

file1 = 'r1_2008.nc';
file2 = 'r2_2008.nc';

display('Done reading');

% Read file 1
ncid=netcdf.open(file1,'NC_NOWRITE'); 
[varname, xtype, dimids, atts] = netcdf.inqVar(ncid,0);
[varname3, xtype3, dimids3, atts3] = netcdf.inqVar(ncid,3);
Pred = ncread(file1,varname3); display('Channel 1 done');

% Read file 2
ncid=netcdf.open(file2,'NC_NOWRITE'); 
[varname, xtype, dimids, atts] = netcdf.inqVar(ncid,0);
[varname3, xtype3, dimids3, atts3] = netcdf.inqVar(ncid,3);
Pred2 = ncread(file2,varname3); display('Channel 2 done');

veg = (Pred2-Pred)./(Pred2+Pred);
%veg = cat(4, Pred, Pred2)

clear Pred; clear Pred2;
%%
% It outputs it as veg (variable name)

notEmpty = [12,13,18,21,23,28,33,36,37,41,43,61,62,63,65,76,84,92,93,94,...
    95,120,122,160,167,170,171,177,179,180,185,186,189,199,200,204,...
    205,206,207,208,209,210,211,215,217,226,227,228,229,230,233,234,238,...
    239,244,245,255,274,276,291,320,321,323,315,360,361];


% build tensor and run tucker
tensor_selected_frames = veg(:,:,notEmpty); 

%Plastic + Shrubs
plastic_and_shrubs = tensor_selected_frames(735:755, 540:560, :);
shrubs = tensor_selected_frames(735:743, 540:560, :);
plastic = tensor_selected_frames(750:755, 540:560, :);

%% Visualize tensor (Chapter 6.1)
clims_veg =  [-0.35  0.35];

figure;
for i = 1:10
    subplot(1,10,i)
    imagesc(plastic_and_shrubs(:,:,(i*7-4)), clims_veg), daspect([1 1 1]);  colormap(bluewhitered);
    title(['t = ' num2str(   notEmpty(i*7-4)   ) ])
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
end
h = colorbar
set(h,'Position',[0.95 0.3 0.01 0.45])


%% Chapter 6.2

% Show the subtensors on a map
figure; clims=[-0.35 0.35]
imagesc(plastic_and_shrubs(:,:,notEmpty(4)),clims); daspect([1 1 1]); colormap(bluewhitered); colorbar; xt = get(gca, 'XTick'); set(gca, 'FontSize', 20);
rectangle('Position',[0.6 0.6 20.8 8], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
rectangle('Position',[0.6 15 20.8 6.4], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H

% Frames used to plot as examples
example_frames = [28,43,84];

% Set min and max values
minVal = -0.35 %min(tensor(:))
maxVal = 0.35
clims=[minVal,maxVal]

% Use Tucker
[Factors_shrubs, G_shrubs, ExplX_shrubs, Reconstr_shrubs] = tucker(shrubs,[8,20,1]);
[Factors_plastic, G_plastic,ExplX_plastic, Reconstr_plastic] = tucker(plastic,[5,20,1]);
[Factors_plastic_and_shrubs, G_plastic_and_shrubs, ExplX_plastic_and_shrubs, Reconstr_plastic_and_shrubs] = tucker(plastic_and_shrubs,[20,20,2]);

% Plot before and after reconstructions
plot_before_and_after(plastic, Reconstr_plastic, notEmpty, example_frames, clims,...
    'plastic. Tucker(30,30,1), rank=1 system assumed')
plot_before_and_after(shrubs, Reconstr_shrubs, notEmpty, example_frames, clims,...
    'shrubs. Tucker(30,25,1), rank=1 system assumed')
plot_before_and_after(plastic_and_shrubs, Reconstr_plastic_and_shrubs, notEmpty, example_frames, clims,...
    'plastic and shrubs. Tucker(90,30,2), rank=2 system assumed')

%% Plot time loadings (Chapter 6.4)  
figure; plot_loadings_chronologically(Factors_plastic, Factors_plastic_and_shrubs, notEmpty, 'r', 'Plastic - Time Loadings')
figure; plot_loadings_chronologically(Factors_shrubs, Factors_plastic_and_shrubs,  notEmpty, 'g', 'Shrubs - Time Loadings')
figure; plot_loadings_chronologically(Factors_plastic_and_shrubs, Factors_plastic_and_shrubs, notEmpty,  'g', 'Plastic and shrubs - Time Loadings')
%NOTE: change ylim inside function manually in last plot


% Plot average NDVI values over time for each region 
figure; plot_loadings_chronologically(plastic_avg_pixels, plastic_avg_pixels, notEmpty, 'r', 'Plastic - Average NDVI over time')
figure; plot_loadings_chronologically(shrubs_avg_pixels, shrubs_avg_pixels, notEmpty, 'g', 'Shrubs - Average NDVI over time')


%% Estimate time rank with time loadings (Chapter 6.4)

figure
for i=1:3
    [Factors, G, ExplX, Reconstr] = tucker(plastic_and_shrubs,[20,20,i]);
    subplot(3,1,i)
    title_ =  strcat('Time rank = ', num2str(i));
    plot_loadings_chronologically(Factors, Factors, notEmpty, 'r', title_)
    
    %correlation = corrcoef(Factors{3})
    %figure; imagesc(correlation); colormap hot;
end

%% Estimate time rank using residual plots (Chapter 6.3)
% Plot original + Reconstructions + Residuals
[Factors1, G1, ExplX1, Reconstr1] = tucker(plastic_and_shrubs, [20,20,1]);
[Factors2, G2, ExplX2, Reconstr2] = tucker(plastic_and_shrubs, [20,20,2]);
[Factors3, G3, ExplX3, Reconstr3] = tucker(plastic_and_shrubs, [20,20,3]);

t = find(notEmpty==361)

clims_error=[-0.15 0.15]
figure
subplot(3,3,1) %Reconstr1
imagesc(plastic_and_shrubs(:,:,t), clims), daspect([1 1 1]); title('Original, t=361','fontsize',16); colormap(bluewhitered); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(3,3,2) %Reconstr2
imagesc(plastic_and_shrubs(:,:,t), clims), daspect([1 1 1]); title('Original, t=361','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(3,3,3) %Reconstr2
imagesc(plastic_and_shrubs(:,:,t), clims), daspect([1 1 1]); title('Original, t=361','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);

subplot(3,3,4) %Reconstr1
imagesc(Reconstr1(:,:,t), clims), daspect([1 1 1]); title('Reconstruction using Tucker(30,30,1)','fontsize',16); colormap(bluewhitered); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(3,3,5) %Reconstr2
imagesc(Reconstr2(:,:,t), clims), daspect([1 1 1]); title('Reconstruction using Tucker(30,30,2)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(3,3,6) %Reconstr2
imagesc(Reconstr3(:,:,t), clims), daspect([1 1 1]); title('Reconstruction using Tucker(30,30,3)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
h1 = colorbar  % xpos, ypos, width, length
set(h1,'Position',[0.93 0.40 0.01 0.51]) %colorbar that fits when fig is not expanded

subplot(3,3,7) %Resid1
imagesc(abs(Reconstr1(:,:,t)-plastic_and_shrubs(:,:,t)), clims_error), daspect([1 1 1]); title('Residuals, Tucker(30,30,1)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(3,3,8) %Resid1
imagesc(abs(Reconstr2(:,:,t)-plastic_and_shrubs(:,:,t)), clims_error), daspect([1 1 1]); title('Residuals, Tucker(30,30,2)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(3,3,9) %Resid2
imagesc(abs(Reconstr3(:,:,t)-plastic_and_shrubs(:,:,t)), clims_error), daspect([1 1 1]); title('After, Tucker(30,30,3)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
h2 = colorbar  % xpos, ypos, width, length
set(h2,'Position',[0.93 0.1 0.01 0.21]) %colorbar that fits when fig is not expanded


%% Plot correlation between time loadings and Average Pixel per Frame (Chapter 6.4)

% Find average pixel in each frame 
plastic_avg_pixels = [];
shrubs_avg_pixels  = [];
plastic_and_shrubs_avg_pixels = [];
no_frames = length(plastic(1,1,:));

for i = 1:no_frames
    temp = nansum(nansum((plastic(:, :, i)))) / sum(sum(~isnan(plastic(:,:,i)),2));
    plastic_avg_pixels = [plastic_avg_pixels,temp];
    
    temp = nansum(nansum((shrubs(:, :, i)))) / sum(sum(~isnan(shrubs(:,:,i)),2));
    shrubs_avg_pixels = [shrubs_avg_pixels,temp];
end

% Plot scatter to show correlation (see user-defined functions)
correlation_plot(plastic_avg_pixels, 'none', 'none', Factors_plastic{3}, 'Plastic area');
correlation_plot(shrubs_avg_pixels, 'none', 'none', Factors_shrubs{3}, 'Shrubs area');
correlation_plot(shrubs_avg_pixels, plastic_avg_pixels, 'none', Factors_plastic_and_shrubs{3}, 'Plastic and shrubs data');
correlation_plot(plastic_avg_pixels, shrubs_avg_pixels, 'none', Factors_plastic_and_shrubs{3}, 'Swap areas');

% Calculate correlation coefficients
corr11 = corr(Factors_plastic_and_shrubs{3}(:,2), shrubs_avg_pixels', 'rows','complete');
corr22 = corr(Factors_plastic_and_shrubs{3}(:,1), shrubs_avg_pixels', 'rows','complete');
corr12 = corr(Factors_plastic_and_shrubs{3}(:,2), plastic_avg_pixels', 'rows','complete');
corr21 = corr(Factors_plastic_and_shrubs{3}(:,1), plastic_avg_pixels', 'rows','complete');


%% Anomaly, last time point is an outlier (not used in thesis)
figure; 
imagesc(plastic_and_shrubs(:,:,66), clims); colormap(bluewhitered); colorbar;
xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);


%% VIDEO (not used in thesis)

%figure; VideoPlot(Reconstr_plastic, Factors_plastic);
%figure; VideoPlot(Reconstr_shrubs, Factors_shrubs);

figure; 
VideoPlot(plastic_and_shrubs, Reconstr_plastic_and_shrubs, Factors_plastic_and_shrubs, 0.3);

figure;
imagesc(plastic_and_shrubs(:,:,66))

%% Reconstruct manually (Chapter 6.4)

%Pick apart the reconstruction
p = 1; q = 1; r = 1;
T = 1;
Reconstr_temp = G_plastic_and_shrubs(p,q,r)*( (Factors_plastic_and_shrubs{1}(:,p)*Factors_plastic_and_shrubs{2}(:,q)') * Factors_plastic_and_shrubs{3}(T,r));

figure; imagesc(Reconstr_temp(:,:,:),clims); colorbar; colormap(bluewhitered);

%% Reconstruct the whole tensor (Chapter 6.4)
I=length(plastic_and_shrubs(:,1,1)); J=length(plastic_and_shrubs(1,:,1)); K=length(plastic_and_shrubs(1,1,:));
P=length((Factors_plastic_and_shrubs{1}(1,:))); Q=length((Factors_plastic_and_shrubs{2}(1,:))); R=length((Factors_plastic_and_shrubs{3}(1,:)));
    
Reconstr_temp = zeros(I, J, K);
Reconstr = zeros(I, J, K);
for t=1:K
    for p=1:length((Factors_plastic_and_shrubs{1}(1,:))) %P
        for q=1:length((Factors_plastic_and_shrubs{2}(1,:))) %Q
            for r=1:length((Factors_plastic_and_shrubs{3}(1,:))) %R
                Reconstr_temp(:,:,t) = G_plastic_and_shrubs(p,q,r)*( (Factors_plastic_and_shrubs{1}(:,p)*Factors_plastic_and_shrubs{2}(:,q)') * Factors_plastic_and_shrubs{3}(t,r));
                Reconstr(:,:,t) = Reconstr(:,:,t) + Reconstr_temp(:,:,t);
            end
        end
    end
end 
figure
imagesc(Reconstr(:,:,1)); colormap(bluewhitered);


%% Reconstruct ONLY using time loading 1 OR 2 (Chapter 6.4)
figure;
for r=1:2
    I=length(plastic_and_shrubs(:,1,1)); J=length(plastic_and_shrubs(1,:,1)); K=length(plastic_and_shrubs(1,1,:));
    Reconstr_temp = zeros(I, J, K);
    Reconstr = zeros(I, J, K);
    for t=1:K
        for p=1:P 
            for q=1:Q 
                %for r=1:R
                    Reconstr_temp(:,:,t) = G_plastic_and_shrubs(p,q,r)*( (Factors_plastic_and_shrubs{1}(:,p)*Factors_plastic_and_shrubs{2}(:,q)') * Factors_plastic_and_shrubs{3}(t,r));
                    Reconstr(:,:,t) = Reconstr(:,:,t) + Reconstr_temp(:,:,t);
                %end
            end
        end
    end 

     for time = 1:10
         if r==1
             subplot(2,10,time)
             imagesc(Reconstr(:,:,(time*7-4)), clims); daspect([1 1 1]); colormap(bluewhitered);
             title( sprintf(['TL 1 \n t = ' num2str(notEmpty(time*7-4)) ] ) ) 
             xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
         end
         if r==2
             subplot(2,10,time+10)
             imagesc(Reconstr(:,:,(time*7-4)), clims); daspect([1 1 1]); colormap(bluewhitered);
             title( sprintf(['TL 2 \n t = ' num2str(notEmpty(time*7-4)) ] ) )
             xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
         end
     end 
end
 h = colorbar
 set(h,'Position',[0.95 0.1 0.01 0.82])

%% Factor map - reconstruct without scaling by time loadings (Chapter 6.4)

%Start by removing the time loadings
new_F = {Factors_plastic_and_shrubs{1}, Factors_plastic_and_shrubs{2}}  %skip time loadings
new_G = G_plastic_and_shrubs(:,:,1)

Reconstr = zeros(I, J);
Reconstr_temp = zeros(I, J);

for p=1:length((new_F{1}(1,:))) %P
    for q=1:length((new_F{2}(1,:))) %Q
            Reconstr_temp = new_G(p,q)*( (new_F{1}(:,p)*new_F{2}(:,q)'));
            Reconstr = Reconstr + Reconstr_temp; 
    end
end

clims = [-0.35 0.35]
figure
imagesc(Reconstr); colormap(bluewhitered); colorbar;  xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
 

%% Plot reconstructions + Time loadings together (Chapter 6.5: Final model)

RANK = 2
[Factors_plastic_and_shrubs, G_plastic_and_shrubs, ExplX_plastic_and_shrubs, Reconstr_plastic_and_shrubs] = tucker(plastic_and_shrubs,[20,20,RANK]);

clims_veg = [-0.35 0.35] 
figure;

for i = 1:10
    % Plot original tensor
    t1 = subplot(4,10,i)
    imagesc(plastic_and_shrubs(:,:,(i*7-4)), clims_veg); daspect([1 1 1]); colormap(bluewhitered);
    title( sprintf(['Original \n t = ' num2str( notEmpty(i*7-4) ) ] ) ) 
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);

    % Plot the reconstruction
    subplot(4,10,i+10)
    imagesc(Reconstr_plastic_and_shrubs(:,:,(i*7-4)), clims_veg); daspect([1 1 1]); colormap(bluewhitered);
    title( 'Reconstr.' ) 
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
    
    % Plot Residuals  USE MONTHS
    subplot(4,10,i+20)
    imagesc(abs((Reconstr_plastic_and_shrubs(:,:,(i*7-4)) -  plastic_and_shrubs(:,:,(i*7-4)))), clims_veg); daspect([1 1 1]); colormap(bluewhitered);
    title('Abs. Res.') 
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
    if (i==4 || i==5 || i==6 || i==7)
        set(gca,'xtick',[])
    end
end

suptitle('Selected frames from the original tensor, the reconstruction, its residuals and corresponding time loadings. ', 'fontsize', 18) 
fudge=2;

h = colorbar
set(h,'Position',[0.95 0.32 0.01 0.56])

ax2 = subplot(4,10,[31:40]) 

axis manual
plot_loadings_chronologically(Factors_plastic_and_shrubs,Factors_plastic_and_shrubs, notEmpty, 'g', 'Time Loadings, Rank 2 Tucker Model')
xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);

for i=1:10
    vline([notEmpty(i*7-4)],'color','k','linestyle',':')
end




%%
%%
%% Plots and calculations for chapter 3 (Data)
 %% count NaN in whole dataset

% Check what percentage total is nan
perc_nan = sum(sum(sum(isnan(Pred(:,:,:))))) / numel(Pred);

% Find all empty frames
all_frames = 1:length(Pred(1,1,:));
not_empty_frames = [];

for i = 1:length(Pred(1,1,:))
   if  sum(sum(~isnan(Pred(:,:,i)))) > 0
       not_empty_frames = [not_empty_frames; i];
   end
end

empty_frames=setdiff(all_frames,not_empty_frames);
perc_empty_frames = length(empty_frames) / length(all_frames);

% Perc. missing data only counting non-empty frames
perc_nan = sum(sum(sum(isnan(Pred(:,:,not_empty_frames))))) / numel(Pred(:,:,not_empty_frames))

%% Now do the same with cut-out in the middle that contains no ocean

% Check what percentage total is nan
perc_nan = sum(sum(sum(isnan(Pred(150:600, 150:700,:))))) / numel(Pred(150:600, 150:700,:))
perc_nan_no_empty_frames = sum(sum(sum(isnan(Pred(150:600, 150:700,not_empty_frames))))) / numel(Pred(150:600, 150:700,not_empty_frames))

%Plot with grid
clims = [0, 250]
figure
imagesc(Pred(:,:,23), clims), daspect([1 1 1]);
colormap hot
colorbar

hold on
rectangle('Position',[150 150 550 450], 'EdgeColor','b') %X,Y,W,H
hold off

%% Just get rid of the ocean!

% Extract a mask for the ocean
ocean_mask = zeros( length(Pred(:,1,1)), length(Pred(1,:,1)) );

%Check if any nan at index for the ocean mask
for x=1:length(Pred(:,1,1))
    for y=1:length(Pred(1,:,1))
        if sum(~isnan(Pred(x,y,:))) > 0
           ocean_mask(x,y) = 1; 
        end
    end
end
%%

% Count NaN where ocean_mask == 1
masked_tensor = zeros( size(Pred));

for t = 1:length(Pred(1,1,:))
    for x = 1: length(Pred(:,1,1))
        for y = 1:length(Pred(1,:,1))
            if ocean_mask(x,y) == 1
                masked_tensor(x,y,t) = Pred(x,y,t);
            else 
                masked_tensor(x,y,t) = 0;
            end 
        end
    end
    
    display(t)
    %masked_tensor(:,:,t) = Pred(:,:,t) .* ocean_mask;
    %need to make NaN=0 where ocean_mask is 0!!!
end

% Calculate percentage NaN within the ocean mask
perc_nan_masked = sum(sum(sum(isnan(masked_tensor(:,:,:))))) / ( sum(sum(ocean_mask)) * 366 )

% Same, but only include non-empty frames
perc_nan_masked_non_empty_frames = sum(sum(sum(isnan(masked_tensor(:,:,not_empty_frames))))) / ( sum(sum(ocean_mask)) * length(not_empty_frames) )

%% Plot ocean mask

figure;
imagesc(ocean_mask); colormap hot;
xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
