%% Read data
clc; clear all; close all;

file1 = 'r1_2008.nc';
file2 = 'r2_2008.nc';

% Read file 1
ncid=netcdf.open(file1,'NC_NOWRITE'); 
[varname, xtype, dimids, atts] = netcdf.inqVar(ncid,0);
[varname3, xtype3, dimids3, atts3] = netcdf.inqVar(ncid,3);
Pred = ncread(file1,varname3); 

% Read file 2
ncid=netcdf.open(file2,'NC_NOWRITE'); 
[varname, xtype, dimids, atts] = netcdf.inqVar(ncid,0);
[varname3, xtype3, dimids3, atts3] = netcdf.inqVar(ncid,3);
Pred2 = ncread(file2,varname3);

% Format NaN's
%data(isnan(data))=0;
Pred(Pred<=0)=NaN; %Find NaN's and change their format for Tucker
Pred2(Pred2<=0)=NaN;

veg = (Pred2-Pred)./(Pred2+Pred);
%veg = cat(4, Pred, Pred2)
clear Pred; clear Pred2;

notEmpty = [4 12 21 23 30 31 33 36 40 43 46 61 62 63 65 84 92 93 94 104 113 ...
    117 120 122 123 167 169 170 171 174 177 179 180 184 185 186 189 194 195 ... 
    197 199 200 203 204 205 206 207 208 209 210 213 215 217 222 226 227 228 ... 
    229 232 233 234 235 236 237 238 244 245 252 257 264 276 280 300 301 315 ...
    318 319 320 321 323 324 328 360 361];

%% Fill in missing data using 3D KNN 

tensor = veg(400:600,300:600,notEmpty);
tensor_filled = inpaintn(tensor);

%% Plot examples before and after filling data with KNN
clims  = [min(min(tensor_filled(:,:,1))), max(max(tensor_filled(:,:,1)))]

figure 
ax1 = subplot(2,2,1)
imagesc(tensor(:,:,40),clims)
colormap hot
colorbar
ax2 = subplot(2,2,2)
imagesc(tensor_filled(:,:,40),clims)
colorbar
ax3 = subplot(2,2,3)
imagesc(tensor(:,:,7),clims)
colorbar
ax4 = subplot(2,2,4)
imagesc(tensor_filled(:,:,7),clims)
colorbar

%% Randomly add missing data

tensor_nan = tensor_filled;

Y = length(tensor_filled(:,1,1));
X = length(tensor_filled(1,:,1));
T = length(tensor_filled(1,1,:));

perc_nan = 0;
i = 0;
while perc_nan<0.05
    Yrand=randi(Y); Xrand=randi(X); Trand=randi(T);
    ymin = max(1, Yrand-1); ymax=min(Y, Yrand+1);
    xmin = max(1, Xrand-1); xmax=min(X, Xrand+1);
    tmin = max(1, Trand-1); tmax=min(T, Trand+1);
    
    tensor_nan(ymin:ymax, xmin:xmax, tmin:tmax) = nan;
    
    % Calculate percentage nan
    perc_nan = sum(sum(sum(isnan(tensor_nan(:,:,:))))) / numel(tensor_nan);
    i=i+1;
    if rem(i,200) == 0
        disp(perc_nan)
    end
end

perc_nan
imagesc(tensor_nan(:,:,1))

%% Run tucker on tensor_nan 
%% and calculate average pixel difference where data was missing

[Factors, G, ExplX, Reconstr] = tucker(tensor_nan,[100,100,2]);

%find indices of all elements in tensor_nan where there is nan
tensor_error = ((sqrt(Reconstr-tensor_filled)).^2).*isnan(tensor_nan);
tensor_avg_error = sum(sum(sum(tensor_error))) / sum(sum(sum(isnan(tensor_nan))));
relative_avg_error = tensor_avg_error / mean2(tensor_filled)

fprintf('The relative average error for a missing pixel: %d.\n', relative_avg_error);
