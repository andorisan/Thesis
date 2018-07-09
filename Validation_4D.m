%% Chapter 5.2
% This includes the 3D vs 4D model comparison

%% Read data
% Selected region in the tensor that contains no missing data 
clc; clear all; close all;

file1 = 'C:\Users\andri\OneDrive\Desktop\Thesis\Matlab Codes\r1_2008.nc'; 
file2 = 'C:\Users\andri\OneDrive\Desktop\Thesis\Matlab Codes\r2_2008.nc';

frames_with_no_nan = [12, 21, 23, 24, 27,28, 31,37, 41,42,61, 62,63, 76,... 
   81,93, 94, 95,120,122,170,171,177, 179,185,186,189, 191,195,199,200,...
   205,207,208,209,210,211,217,220, 228,230,233, 234, 237,238,257,278,...
   301,315,321,326,332,354,360]; %see getFrames() in Validation.m

% Read file 1
ncid=netcdf.open(file1,'NC_NOWRITE'); 
[varname, xtype, dimids, atts] = netcdf.inqVar(ncid,0);
[varname3, xtype3, dimids3, atts3] = netcdf.inqVar(ncid,3);
Pred = ncread(file1,varname3); 
c1 = Pred(491:520,531:560,frames_with_no_nan);
clear Pred; display('Channel 1 done')

% Read file 2
ncid=netcdf.open(file2,'NC_NOWRITE'); 
[varname, xtype, dimids, atts] = netcdf.inqVar(ncid,0);
[varname3, xtype3, dimids3, atts3] = netcdf.inqVar(ncid,3);
Pred2 = ncread(file2,varname3);
c2 = Pred2(491:520,531:560,frames_with_no_nan);
clear Pred2; display('Channel 2 done')

veg = (c2-c1)./(c2+c1);
c12 = cat(4, c1, c2);  %c12 = 4d tensor with both channels


%% Compare 3D vs 4D models (ch 5.2)

rmse_list_4D_veg=[]
rmse_list_4D_veg_syst=[]
%rmse_list_4D_c1=[]; rmse_list_4D_c1_syst=[];
%rmse_list_4D_c2=[]; rmse_list_4D_c2_syst=[];
rmse_list_3D = []; rmse_list_3D_syst = [];
perc_nan_list       = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99];

for i=1:length(perc_nan_list)
    %______________________________________________________________________
    % 4D CASE - use both channels
    %Add missing data
    fprintf('Iteration: %d -a.\n',i);
    tensor4D_nan = AddMissingData_4D(c12, perc_nan_list(i), 0, 'equal_');  %setting = 'equal_' or 'random'
    tensor4D_nan_syst = AddMissingData_4D(c12, perc_nan_list(i), 2, 'equal_');  %setting = 'equal_' or 'random'
    
    % Random case
    [Factors, G, ExplX, Reconstr] = tucker(tensor4D_nan,[30,30,2,2]); 
    rmse_veg = Calc_RMSE_4D(tensor4D_nan(:,:,:,1), Reconstr, veg)
    %rmse_c1 = Calc_RMSE_4D( tensor4D_nan(:,:,:,1), Reconstr(:,:,:,1), c1);
    %rmse_c2 = Calc_RMSE_4D( tensor4D_nan(:,:,:,2), Reconstr(:,:,:,2), c2);
    
    rmse_list_4D_veg = [rmse_list_4D_veg; rmse_veg]; %laga listanofn
    %rmse_list_4D_c1 = [rmse_list_4D_c1; rmse_c1];
    %rmse_list_4D_c2 = [rmse_list_4D_c2; rmse_c2];

    % Systematic case%
    [Factors, G, ExplX, Reconstr] = tucker(tensor4D_nan_syst,[30,30,2,2]); 
    %Calculate RMSE 4D-case, channel 1 and 2
    rmse_veg_syst = Calc_RMSE_4D( tensor4D_nan_syst(:,:,:,1), Reconstr, veg);
    %rmse_c1_syst = Calc_RMSE_4D( tensor4D_nan_syst(:,:,:,1), Reconstr(:,:,:,1), c1);
    %rmse_c2_syst = Calc_RMSE_4D( tensor4D_nan_syst(:,:,:,2), Reconstr(:,:,:,2), c2)
    
    rmse_list_4D_veg_syst = [rmse_list_4D_veg_syst; rmse_veg_syst];
    %rmse_list_4D_c1_syst = [rmse_list_4D_c1_syst; rmse_c1_syst];
    %rmse_list_4D_c2_syst = [rmse_list_4D_c2_syst; rmse_c2_syst];
    
    %______________________________________________________________________
    % 3D CASE - compress channels before running the model
    fprintf('Iteration: %d -b.\n',i);
    tensor3D_nan = AddMissingData(veg, perc_nan_list(i), 0);
    tensor3D_nan_syst = AddMissingData(veg, perc_nan_list(i), 4);

    %Random pixels case
    [Factors_3d, G_3d, ExplX_3d, Reconstr_3d] = tucker(tensor3D_nan,[30,30,2]);
    rmse = Calc_RMSE_4D(tensor3D_nan, Reconstr_3d, veg);
    rmse_list_3D = [rmse_list_3D; rmse];
    
    %Systematic case
    [Factors, G, ExplX, Reconstr_3d] = tucker(tensor3D_nan_syst,[30,30,2]);
    rmse_syst = Calc_RMSE_4D(tensor3D_nan_syst, Reconstr_3d, veg);
    rmse_list_3D_syst = [rmse_list_3D_syst; rmse_syst];
end

figure
ax1 = subplot(1,2,1)
plot(perc_nan_list*100, rmse_list_4D_veg, 'b')
hold on
%plot(perc_nan_list, rmse_list_4D_c2, 'g')
plot(perc_nan_list*100, rmse_list_3D, 'r')
legend({'4-way Model - Both channels used in the model', '3-way Model - Channels compressed'}, 'fontsize', 14)


xlabel('Percentage data missing', 'fontsize', 16)
ylabel('Average RRMSE', 'fontsize', 16)
ylim(ax1, [0, 1])
title('Missing data pixels added randomly', 'fontsize', 16)
xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
hold off

ax2 = subplot(1,2,2)
plot(perc_nan_list*100, rmse_list_4D_veg_syst, 'b')
hold on
%plot(perc_nan_list, rmse_list_4D_c2_syst, 'g')
plot(perc_nan_list*100, rmse_list_3D_syst, 'r')
legend({'4-way Model - Both channels used in the model', '3-way Model - Channels Compressed'}, 'fontsize', 14)
xlabel('Percentage data missing', 'fontsize', 16)
ylabel('Average RRMSE', 'fontsize', 16)
ylim(ax2, [0, 1])
title('Systematic missing data', 'fontsize', 16)
xt = get(gca, 'XTick'); set(gca, 'FontSize', 16);
hold off

%suptitle('RRMSE for randomly added missing data with and without compressing channels', 'fontsize', 18)
suptitle('RRMSE for overlapping missing data with and without compressing channels', 'fontsize', 18)



