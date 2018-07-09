%% Chapter 5
% This includes all the code for the Validation chapter, except for the 3D
% vs 4D comparison.


%% Read data
clc; clear all; close all;

file1 = 'C:\Users\andri\OneDrive\Desktop\Thesis\Matlab Codes\r1_2008.nc'; 
file2 = 'C:\Users\andri\OneDrive\Desktop\Thesis\Matlab Codes\r2_2008.nc';
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

%clear Pred; clear Pred2;


%% Chapter 5.1
% Warning: This takes a very long time to run!

% Select subset of data and prepare for analysis
% Input (parameters to change): tensor, frames_type = 'real_data'/'simulated', noise = 'no_noise_'/'add noise'
frames = get_frames(veg, 'real_data', 'no_noise_');   %(491:520,531:560,23);

error_setting = 'RMSE';  %Select error measurement: RMSE or CORR

% Plot with noise
%figure; clims_pos=[0 0.35]
%imagesc(frames(:,:,1),clims_pos); daspect([1 1 1]); colormap jet; colorbar; 
%xt = get(gca, 'XTick'); set(gca, 'FontSize', 16); %make ticks bigger!

% Plot missing data examples (visualization)
tensor_nan = AddMissingData(frames, 0.9, 2);

% Add missing data and calculate average relative pixel error for the reconstruction
perc_nan_list = [0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99];

% Tucker model with EM algorithm
avg_rel_error_list            = [];
avg_rel_error_list_syst       = [];  % missing data is systematic instead of random (bigger blocks)
% Tucker model with single imputation (mean)
avg_rel_error_list_imp        = [];
avg_rel_error_list_syst_imp   = [];
% LMLRA model
avg_rel_error_list_lmlra      = [];
avg_rel_error_list_syst_lmlra = [];
% Always guess the mean, naive benchmark approach
avg_rel_error_list_naive      = [];  
avg_rel_error_list_syst_naive = [];


for i=1:length(perc_nan_list)    
    % Add missing data_____________________________________________________
    frames_missing_temp = AddMissingData(frames, perc_nan_list(i), 0);
    
    % Calculate average relative pixel error (ARPE)
    if perc_nan_list(i) == 0
        avg_rel_error_temp = 0;
        avg_rel_error_imp_temp = 0;
        avg_rel_error_lmlra_temp = 0;
        avg_rel_error_naive_temp = 0;
    else 
        avg_rel_error_temp = Calc_Av_Rel_Error(frames, frames_missing_temp, 'tucker', [30,30,2], error_setting); %plot commands commented out
        display(i); display('a...out of 16 iterations'); %print how far we are
        if perc_nan_list(i) < 0.95 %because these functions break when there is this much missing data
            avg_rel_error_imp_temp = Calc_Av_Rel_Error(frames, frames_missing_temp, 'imptuc', [30,30,2], error_setting);
            avg_rel_error_lmlra_temp = Calc_Av_Rel_Error(frames, frames_missing_temp, 'lmlra_', [30,30,2], error_setting);
        else 
            avg_rel_error_imp_temp = NaN;
            avg_rel_error_lmlra_temp = NaN;
        end 
        avg_rel_error_naive_temp = Calc_Av_Rel_Error(frames, frames_missing_temp, 'naive_', [30,30,2], error_setting);
    end 
    avg_rel_error_list       = [avg_rel_error_list; avg_rel_error_temp]; 
    avg_rel_error_list_imp   = [avg_rel_error_list_imp; avg_rel_error_imp_temp];
    avg_rel_error_list_lmlra = [avg_rel_error_list_lmlra; avg_rel_error_lmlra_temp]; 
    avg_rel_error_list_naive = [avg_rel_error_list_naive; avg_rel_error_naive_temp];
    
    % Add missing data systematically______________________________________
    frames_missing_syst_temp = AddMissingData(frames, perc_nan_list(i), 2);
    
    if perc_nan_list(i) == 0
        avg_rel_error_syst_temp       = 0;
        avg_rel_error_syst_imp_temp   = 0;
        avg_rel_error_syst_lmlra_temp = 0;
        avg_rel_error_syst_naive_temp = 0;
    else 
        avg_rel_error_syst_temp = Calc_Av_Rel_Error(frames, frames_missing_syst_temp, 'tucker', [30,30,2], error_setting);
        display(i); display('b...out of 16 iterations');
        if perc_nan_list(i) < 0.95
            avg_rel_error_syst_imp_temp = Calc_Av_Rel_Error(frames, frames_missing_syst_temp, 'imptuc', [30,30,2], error_setting);
            avg_rel_error_syst_lmlra_temp = Calc_Av_Rel_Error(frames, frames_missing_syst_temp, 'lmlra_', [30,30,2], error_setting);
        else 
            avg_rel_error_syst_imp_temp = NaN;
            avg_rel_error_syst_lmlra_temp = NaN;
        end 
        
        
        avg_rel_error_syst_naive_temp = Calc_Av_Rel_Error(frames, frames_missing_syst_temp, 'naive_', [30,30,2], error_setting);
    end
    avg_rel_error_list_syst = [avg_rel_error_list_syst; avg_rel_error_syst_temp];
    avg_rel_error_list_syst_imp = [avg_rel_error_list_syst_imp; avg_rel_error_syst_imp_temp];
    avg_rel_error_list_syst_lmlra = [avg_rel_error_list_syst_lmlra; avg_rel_error_syst_lmlra_temp];
    avg_rel_error_list_syst_naive = [avg_rel_error_list_syst_naive; avg_rel_error_syst_naive_temp];
end%_______________________________________________________________________


figure
ax1 = subplot(1,2,1)
plot(perc_nan_list*100, avg_rel_error_list_naive, 'k'); hold on;
plot(perc_nan_list*100, avg_rel_error_list_imp, 'c'); 
plot(perc_nan_list*100, avg_rel_error_list, 'b');
plot(perc_nan_list*100, avg_rel_error_list_lmlra, 'r')

title('Missing data pixels added randomly','fontsize',16)
xlabel('Percentage data missing','fontsize',16)
%ylabel('Average correlation', 'fontsize', 16)
ylabel('Average RRMSE','fontsize',16)
%ylim(ax1, [0.25,0.55]) %with noise
%ylim(ax1, [0,0.25]) %without noise
ylim(ax1, [0, 0.6])
hold off
legend(ax1,{'Naive Approach', 'Single Imp. Tucker(30,30,2)', 'EM Tucker(30,30,2)', 'LMLRA(30,30,2)' }, 'Location','northwest','fontsize',14)
xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);

ax2 = subplot(1,2,2)
plot(perc_nan_list*100, avg_rel_error_list_syst_naive, 'k'); hold on;
plot(perc_nan_list*100, avg_rel_error_list_syst_imp, 'c');
plot(perc_nan_list*100, avg_rel_error_list_syst, 'b'); 
plot(perc_nan_list*100, avg_rel_error_list_syst_lmlra, 'r')

title('Systematic missing data','fontsize',16) %(big patches added)
xlabel('Percentage data missing','fontsize',16)
%ylabel('Average correlation', 'fontsize', 16)
ylabel('Average RRMSE','fontsize',16)
%ylim(ax1, [0.25,0.55]) %with noise
%ylim(ax1, [0,0.25]) %without noise
ylim(ax2, [0, 0.6])
hold off
legend(ax2,{'Naive Approach', 'Single Imp. Tucker(30,30,2)', 'EM Tucker(30,30,2)', 'LMLRA(30,30,2)'}, 'Location','northwest','fontsize',14)
xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);


%suptitle('RRMSE for missing data','fontsize',20) %no noise
%suptitle('RRMSE for missing data in noisy dataset','fontsize',20) %noise
suptitle('RRMSE for missing data using a subset of Iberian Dataset','fontsize',20)
%suptitle('Average correlation between frames in the original tensor and the reconstructed tensor', 'fontsize', 20)




%% Plot example of real data frames
%use as standard for veg. ratio: clims =  -0.1000    0.36
minVal = -0.35; %min(tensor(:));
maxVal = 0.35%max(plastic_and_shrubs(:)); %max(forest_and_crops(:));
clims=[minVal,maxVal];

slice = frames(:,:,:);

figure
subplot(2,3,1) 
imagesc(slice(:,:,1), clims), daspect([1 1 1]); title('Frame: 1','fontsize',16);
colormap(bluewhitered);

subplot(2,3,2)
imagesc(slice(:,:,11), clims), daspect([1 1 1]); title('Frame: 11','fontsize',16);

subplot(2,3,3)
imagesc(slice(:,:,21), clims), daspect([1 1 1]); title('Frame: 21','fontsize',16);

subplot(2,3,4)
imagesc(slice(:,:,31), clims), daspect([1 1 1]); title('Frame: 31','fontsize',16);

subplot(2,3,5)
imagesc(slice(:,:,41), clims), daspect([1 1 1]); title('Frame: 41','fontsize',16);

subplot(2,3,6)
imagesc(slice(:,:,51), clims), daspect([1 1 1]); title('Frame: 51','fontsize',16);

% colorbar 
h = colorbar
set(h,'Position',[0.95 0.1 0.01 0.82])




%% Chapter 5.3: RANK APPROXIMATION

sample_frame = veg(491:520,531:560,23);

frame_copies = [];
for i=1:10
    frames_copies(:,:,i) = sample_frame;
end


%% Simulate the effect of changing x,y ranks

% Plot before and after adding values to pixels
clims = [0 0.35]
figure
subplot(2,3,1)
imagesc(frames_copies(:,:,1), clims), daspect([1 1 1]); title('Before'); colormap jet;%colormap(bluewhitered);
title('Before', 'fontsize', 20)
rank_list = [1 2 3 5 30]

for i = 2:6
    [Factors, G, ExplX, Reconstr] = tucker(frames_copies, [rank_list(i-1),rank_list(i-1),1]);
    subplot(2,3,i)
    imagesc(Reconstr(:,:,1), clims), daspect([1 1 1]); 
    title(['Reconstruction,' 'spatial ranks = ', num2str(rank_list(i-1))], 'fontsize', 20)
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 16); %make ticks bigger!
end
h = colorbar
set(h,'Position',[0.935 0.1 0.01 0.82]) %same colorbar for 2 rows


legend(ax2,{'Naive Approach', 'Single Imp. Tucker(30,30,2)', 'EM Tucker(30,30,2)', 'LMLRA(30,30,2)'}, 'Location','northwest','fontsize',14)



%%
%%
%% Chapter 5.3.1: RANK 1 SIMULATION

tensor1 = frames_copies;

times_to_add_list = [3, 6, 9];
values_to_add = [0.1,0.15,0.2];

v = 1;

% Add values to all pixels (Rank 1 system) in certain time points
for t = 1:length(tensor1(1,1,:))
    for x = 1:length(tensor1(:,1,1))
        for y = 1:length(tensor1(1,:,1))
        
            if sum(ismember(t, times_to_add_list)) == 1;
                tensor1(x,y,t) = tensor1(x,y,t) + values_to_add(v);    
            end
        end
    end
    
    if sum(ismember(t, times_to_add_list)) == 1;
        v = v+1;   
    end
end

[Factors, G, ExplX, Reconstr] = tucker(tensor1, [29,29,1]);
[Factors2, G2, ExplX2, Reconstr2] = tucker(tensor1, [30,30,2]);

clims =  [-0.35 0.35];


% Plot before and after adding values to pixels - show all 10 frames!
figure;
for t = 1:10
    subplot(2,10,t)
    imagesc(frames_copies(:,:,t), clims), daspect([1 1 1]); 
    colormap(bluewhitered);
    title( sprintf(['Before, \n t = ' num2str(t) ] ) ) 
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); %make ticks bigger!
    
    subplot(2,10,t+10)
    imagesc(tensor1(:,:,t), clims), daspect([1 1 1]); 
    title( sprintf(['After, \n t = ' num2str(t) ] ) ) 
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); %make ticks bigger!
    
    if ismember(t,times_to_add_list) == 1
        hold on
        rectangle('Position',[1 1 29 29], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
        hold off
    end
end
h = colorbar
set(h,'Position',[0.935 0.16 0.01 0.73]) %same colorbar for 2 rows


% Plot tensor + reconstr. + residuals - all frames!
figure;
for t = 1:10
    subplot(3,10,t)
    imagesc(tensor1(:,:,t), clims), daspect([1 1 1]); 
    colormap(bluewhitered);
    title( sprintf(['Tensor, \n t = ' num2str(t) ] ) ) 
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); %make ticks bigger!
    
    subplot(3,10,t+10)
    imagesc(Reconstr2(:,:,t), clims), daspect([1 1 1]); 
    title( sprintf(['Reconstr., \n t = ' num2str(t) ] ) ) 
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); %make ticks bigger!
    
    if ismember(t,times_to_add_list) == 1
        hold on
        rectangle('Position',[1 1 29 29], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
        hold off
    end
    
    subplot(3,10,t+20)
    imagesc(tensor1(:,:,t)-Reconstr2(:,:,t), clims), daspect([1 1 1]); 
    title( sprintf(['Residuals, \n t = ' num2str(t) ] ) ) 
    xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); %make ticks bigger!
    
    if ismember(t,times_to_add_list) == 1
        hold on
        rectangle('Position',[1 1 29 29], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
        hold off
    end
end
h = colorbar
set(h,'Position',[0.935 0.16 0.01 0.73]) %same colorbar for 2 rows


% Plot factors
figure
ax=subplot(2,1,1)
plot(Factors{3}); title('Time loadings for the rank 1 model', 'fontsize', 20); 
xt = get(gca, 'XTick'); set(gca, 'FontSize', 16); %make ticks bigger!
legend(ax,{'Time loading 1'}, 'fontsize',14)
ax2=subplot(2,1,2)
plot(Factors2{3}); title('Time loadings for the rank 2 model', 'fontsize', 20)
xt = get(gca, 'XTick'); set(gca, 'FontSize', 16); %make ticks bigger!
legend(ax2,{'Time loading 1', 'Time loading 2'}, 'fontsize',14)

% Correlate mean pixels values in frames and time loadings
frame_averages = [];
for t = 1:length(tensor1(1,1,:))
    frame_averages = [frame_averages; mean2(tensor1(:,:,t))];
end

addpath(genpath('C:\Users\andri\OneDrive\Desktop\Thesis\Matlab Codes'))


correlation_plot(frame_averages', frame_averages, frame_averages, Factors{3}, 'Rank 1')


correlation_plot(frame_averages', frame_averages', Factors2{3}, 'Rank 2')
correl1 = corr(Factors2{3}(:,1), frame_averages)
correl2 = corr(Factors2{3}(:,2), frame_averages)

correl1 = corr(Factors2{3}(:,2), frame_averages)
correl2 = corr(Factors2{3}(:,1), frame_averages)



%%
%%
%% Chapter 5.3.2: RANK 2 SIMULATION

tensor2 = frames_copies;

times_to_add_list  = [3, 6, 9];
values_to_add  = [0.1,0.15,0.2];

for i = 1:length(times_to_add_list)
    tensor2(1:10, :, times_to_add_list(i)) = tensor2(1:10, :, times_to_add_list(i)) + values_to_add(i);
end


% Plot before and after adding values to pixels - show all 10 frames!
figure;
for t = 1:10
    subplot(2,10,t)
    imagesc(frames_copies(:,:,t), clims), daspect([1 1 1]); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
    colormap(bluewhitered);
    title(sprintf('Before,\n t = %d', t))
    
    subplot(2,10,t+10)
    imagesc(tensor2(:,:,t), clims), daspect([1 1 1]); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); 
    title(sprintf('After,\n t = %d', t))
    
    
    if ismember(t,times_to_add_list) == 1
        hold on
        rectangle('Position',[1 1 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
        hold off
    end
end
h = colorbar
set(h,'Position',[0.935 0.16 0.01 0.73]) %same colorbar for 2 rows
xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); 

% Plot original + Reconstructions + Residuals
[Factors1, G1, ExplX1, Reconstr1] = tucker(tensor2, [30,30,1]);
[Factors2, G2, ExplX2, Reconstr2] = tucker(tensor2, [30,30,2]);


clims_error=[-0.15 0.15]
figure
subplot(2,3,1) %originial
imagesc(tensor2(:,:,6), clims), daspect([1 1 1]); title('Tensor frame','fontsize',16); colormap(bluewhitered); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(2,3,2) %Reconstr1
imagesc(Reconstr1(:,:,6), clims), daspect([1 1 1]); title('Reconstruction using Tucker(30,30,1)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(2,3,3) %Reconstr2
imagesc(Reconstr2(:,:,6), clims), daspect([1 1 1]); title('Reconstruction using Tucker(30,30,2)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);

h1 = colorbar  % xpos, ypos, width, length
set(h1,'Position',[0.93 0.57 0.01 0.37]) %colorbar that fits when fig is not expanded
subplot(2,3,5) %Resid1
imagesc(abs(Reconstr1(:,:,6)-tensor2(:,:,6)), clims_error), daspect([1 1 1]); title('Residuals, Tucker(30,30,1)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
subplot(2,3,6) %Resid2
imagesc(abs(Reconstr2(:,:,6)-tensor2(:,:,6)), clims_error), daspect([1 1 1]); title('Residuals, Tucker(30,30,2)','fontsize',16); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
h2 = colorbar  % xpos, ypos, width, length
set(h2,'Position',[0.93 0.1 0.01 0.35]) %colorbar that fits when fig is not expanded

% Plot factors
figure
ax1 = subplot(2,1,1); plot(Factors1{3}); ylim(ax1, [min(Factors1{3}), max(Factors1{3})+0.005] ); 
title('Time loadings for the rank 1 model','fontsize',12);
legend(ax1,{'Time loading 1'}, 'fontsize',12)
ax2 = subplot(2,1,2); plot(Factors2{3}); 
title('Time loadings for the rank 2 model', 'fontsize', 12);
legend(ax2,{'Time loading 1', 'Time loading 2'}, 'fontsize',12)

% Correlate mean pixels values in frames and time loadings
frame_averages = [];
frame_averages_upper = [];
frame_averages_lower = [];
for t = 1:length(tensor2(1,1,:))
    frame_averages = [frame_averages; mean2(tensor2(:,:,t))];
    frame_averages_upper = [frame_averages_upper; mean2(tensor2(1:10,  :,t))];
    frame_averages_lower = [frame_averages_lower; mean2(tensor2(11:end,:,t))];
end

%Rank 1 correlation plot
correlation_plot(frame_averages', 'no_input','no_input', Factors1{3}, 'Rank 1')
correl = corr(Factors1{3}, frame_averages)

%Rank 2 correlation plot
correlation_plot(frame_averages_upper', frame_averages_lower', 'no_input', Factors2{3}, 'Rank 2')


%%% Show contribution of each time loading %%%
%Reconstruct ONLY using time loading 1 OR 2
figure;
P=length((Factors2{1}(1,:))); Q=length((Factors2{2}(1,:))); R=length((Factors2{3}(1,:)));
for r=1:2
    I=length(tensor2(:,1,1)); J=length(tensor2(1,:,1)); K=length(tensor2(1,1,:));
    Reconstr_temp = zeros(I, J, K);
    Reconstr = zeros(I, J, K);
    for t=1:K
        for p=1:P 
            for q=1:Q 
                %for r=1:R
                    Reconstr_temp(:,:,t) = G2(p,q,r)*( (Factors2{1}(:,p)*Factors2{2}(:,q)') * Factors2{3}(t,r));
                    Reconstr(:,:,t) = Reconstr(:,:,t) + Reconstr_temp(:,:,t);
                %end
            end
        end
    end 

     for time = 1:10
         if r==1
             clims = [-0.35 0.35]
             subplot(2,10,time)
             imagesc(Reconstr(:,:,(time)), clims); daspect([1 1 1]); colormap(bluewhitered); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
             
             %add square to show where change is
             if ismember(time,times_to_add_list) == 1
                hold on
                rectangle('Position',[1 1 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
                hold off
             end
             
             title(['TL 1, t = ' num2str(time) ])
         end
         
         if r==2
             clims = [-0.35 0.35]
             subplot(2,10,time+10)
             imagesc(Reconstr(:,:,(time)), clims); daspect([1 1 1]); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
             
             %add square to show where change is
             if ismember(time,times_to_add_list) == 1
                hold on
                rectangle('Position',[1 1 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
                hold off
             end
             
             title(['TL 2, t = ' num2str(time) ])
         end
         
         
     end 
end
h = colorbar
set(h,'Position',[0.935 0.16 0.01 0.73]) %same colorbar for 2 rows


%%
%%
%% Chapter 5.3.3: Rank 3 simulation
                
tensor3 = frames_copies;

values_to_add  = [0.1,0.15,0.2];
values_to_add2  = [-0.20, -0.15, -0.125, -0.15, -0.20];

times_to_add_list  = [3, 6, 9];
times_to_add_list2 = [2, 4, 6, 8, 10];

for i = 1:length(times_to_add_list)
    tensor3(1:10, :, times_to_add_list(i)) = tensor3(1:10, :, times_to_add_list(i)) + values_to_add(i);
end
for i = 1:length(times_to_add_list2)
    tensor3(21:end, :, times_to_add_list2(i)) = tensor3(21:end, :, times_to_add_list2(i)) + values_to_add2(i);
end

% Plot before and after adding values to pixels - show all 10 frames!
figure;
for t = 1:10
    subplot(2,10,t)
    imagesc(frames_copies(:,:,t), clims), daspect([1 1 1]); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
    colormap(bluewhitered);
    title(sprintf('Before,\n t = %d', t))
    
    subplot(2,10,t+10)
    imagesc(tensor3(:,:,t), clims), daspect([1 1 1]); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
    title(sprintf('After,\n t = %d', t))
    
    if ismember(t,times_to_add_list) == 1
        hold on
        rectangle('Position',[1 1 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
        hold off
    end
    if ismember(t,times_to_add_list2) == 1
        hold on
        rectangle('Position',[1 21 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
        hold off
    end
    
end
h = colorbar
set(h,'Position',[0.935 0.16 0.01 0.73]) %same colorbar for 2 rows

% Plot original + Reconstructions + Residuals
[Factors1, G1, ExplX1, Reconstr1] = tucker(tensor3, [30,30,1]);
[Factors2, G2, ExplX2, Reconstr2] = tucker(tensor3, [30,30,2]);
[Factors3, G3, ExplX3, Reconstr3] = tucker(tensor3, [30,30,3]);

clims_error=[-0.15 0.15]
figure
subplot(2,3,1) %Reconstr1
imagesc(Reconstr1(:,:,6), clims), daspect([1 1 1]); title('Reconstruction using Tucker(30,30,1)','fontsize',20); colormap(bluewhitered); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
subplot(2,3,2) %Reconstr2
imagesc(Reconstr2(:,:,6), clims), daspect([1 1 1]); title('Reconstruction using Tucker(30,30,2)','fontsize',20); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
subplot(2,3,3) %Reconstr2
imagesc(Reconstr3(:,:,6), clims), daspect([1 1 1]); title('Reconstruction using Tucker(30,30,3)','fontsize',20); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
h1 = colorbar  % xpos, ypos, width, length
set(h1,'Position',[0.93 0.57 0.01 0.37]) %colorbar that fits when fig is not expanded

subplot(2,3,4) %Resid1
imagesc(abs(Reconstr1(:,:,6)-tensor3(:,:,6)), clims_error), daspect([1 1 1]); title('Residuals, Tucker(30,30,1)','fontsize',20); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
subplot(2,3,5) %Resid1
imagesc(abs(Reconstr2(:,:,6)-tensor3(:,:,6)), clims_error), daspect([1 1 1]); title('Residuals, Tucker(30,30,2)','fontsize',20); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
subplot(2,3,6) %Resid2
imagesc(abs(Reconstr3(:,:,6)-tensor3(:,:,6)), clims_error), daspect([1 1 1]); title('Residuals, Tucker(30,30,3)','fontsize',20); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
h2 = colorbar  % xpos, ypos, width, length
set(h2,'Position',[0.93 0.1 0.01 0.35]) %colorbar that fits when fig is not expanded

% Plot factors
figure
ax1=subplot(3,1,1); plot(Factors1{3}); ylim(ax1, [min(Factors1{3}), max(Factors1{3})+0.005] ); 
title('Time loadings for the rank 1 model'); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
legend(ax1,{'Time loading 1'}, 'fontsize',12)

ax2=subplot(3,1,2); plot(Factors2{3}); 
title('Time loadings for the rank 2 model'); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
legend(ax2,{'Time loading 1', 'Time loading 2'}, 'fontsize',12)

ax3=subplot(3,1,3); plot(Factors3{3}); 
title('Time loadings for the rank 3 model'); xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
legend(ax3,{'Time loading 1', 'Time loading 2', 'Time loading 3'}, 'fontsize',12)



% Correlate mean pixels values in frames and time loadings
frame_averages_middle = [];
frame_averages_upper = [];
frame_averages_lower = [];
for t = 1:length(tensor3(1,1,:))
    frame_averages_upper = [frame_averages_upper; mean2(tensor3(1:10,  :,t))];
    frame_averages_middle = [frame_averages_middle; mean2(tensor3(11:20,:,t))];
    frame_averages_lower = [frame_averages_lower; mean2(tensor3(21:end,:,t))];
end

%Rank 3 correlation plot
addpath(genpath('C:\Users\andri\OneDrive\Desktop\Thesis\Matlab Codes'))
correlation_plot(frame_averages_upper', frame_averages_middle', frame_averages_lower', Factors3{3}, 'Rank 3')

correl1 = corr(Factors3{3}(:,1), frame_averages_upper)
correl2 = corr(Factors3{3}(:,2), frame_averages_middle)
correl3 = corr(Factors3{3}(:,3), frame_averages_lower)

correl13 = corr(Factors3{3}(:,1), frame_averages_lower)
correl31 = corr(Factors3{3}(:,2), frame_averages_upper)


%%% Show contribution of each time loading %%%
%Reconstruct ONLY using time loading 1 OR 2
figure;
P=length((Factors3{1}(1,:))); Q=length((Factors3{2}(1,:))); R=length((Factors3{3}(1,:)));
for r=1:3
    I=length(tensor3(:,1,1)); J=length(tensor3(1,:,1)); K=length(tensor3(1,1,:));
    Reconstr_temp = zeros(I, J, K);
    Reconstr = zeros(I, J, K);
    for t=1:K
        for p=1:P 
            for q=1:Q 
                %for r=1:R
                    Reconstr_temp(:,:,t) = G3(p,q,r)*( (Factors3{1}(:,p)*Factors3{2}(:,q)') * Factors3{3}(t,r));
                    Reconstr(:,:,t) = Reconstr(:,:,t) + Reconstr_temp(:,:,t);
                %end
            end
        end
    end 

     for time = 1:10
         if r==1
             clims = [-0.35 0.35]
             subplot(3,10,time)
             imagesc(Reconstr(:,:,(time)), clims); daspect([1 1 1]); colormap(bluewhitered); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
             
             %Add squares to show where changes were made
             if ismember(time,times_to_add_list) == 1
                hold on
                rectangle('Position',[1 1 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
                hold off
             end
             if ismember(time,times_to_add_list2) == 1
                hold on
                rectangle('Position',[1 21 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
                hold off
             end   
             title(['TL 1, t = ' num2str(time) ])
         end
         
         
         if r==2
             clims = [-0.35 0.35]
             subplot(3,10,time+10)
             imagesc(Reconstr(:,:,(time)), clims); daspect([1 1 1]);  xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);

             % Add squares to show where changes were made
             if ismember(time,times_to_add_list2) == 1
                hold on
                rectangle('Position',[1 21 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
                hold off
             end
             title(['TL 2, t = ' num2str(time) ])
         end
         
         if r==3
             clims = [-0.35 0.35]
             subplot(3,10,time+20)
             imagesc(Reconstr(:,:,(time)), clims); daspect([1 1 1]); xt = get(gca, 'XTick'); set(gca, 'FontSize', 14); 
             
             % Add squares to show where changes were made
             if ismember(time,times_to_add_list) == 1
                hold on
                rectangle('Position',[1 1 29 9], 'LineWidth',2,'EdgeColor', 'blue') %X,Y,W,H
                hold off
             end
             title(['TL 3, t = ' num2str(time) ])
         end
     end 
end
h2 = colorbar  % xpos, ypos, width, length
set(h2,'Position',[0.93 0.15 0.01 0.75]) %colorbar that fits when fig is not expanded
xt = get(gca, 'XTick'); set(gca, 'FontSize', 14);
             

