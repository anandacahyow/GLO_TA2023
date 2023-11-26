clear;clc;close all

% import data
well = importdata('well12_revised.csv')
%readtable('well2.csv')
head(readtable('well12_revised.csv'))

% data preparation
% date = well.textdata(2:end,1);
% date = datenum(date);

glir1 = well.data(:,1);
qo1 = well.data(:,2);
qt1 = well.data(:,5);
wc1 = well.data(:,3);

glir2 = well.data(:,6);
qo2 = well.data(:,7);
qt2 = well.data(:,10);
wc2 = well.data(:,8);

t = [0:length(glir1)-1];

n_sampling = 0.1;

plot(t,qo1)
hold on
%plot(t,glir1)
plot(t,qt1)
plot(t,wc1)
%plot(t,glir2)
plot(t,qo2)
plot(t,qt2)
plot(t,wc2)

%% System Out
glir11 = out.well1(:,1);
qo11 = out.well1(:,2);
qt11 = out.well1(:,3);
wc11 = out.well1(:,4);

glir22 = out.well2(:,1);
qo22 = out.well2(:,2);
qt22 = out.well2(:,3);
wc22 = out.well2(:,4);

index = [1:length(glir11)]';

T = table(index,glir11,qo11,qt11,wc11,glir22,qo22,qt22,wc22);
head(readtable('well12_rev'))
head(T)

S = table2struct(T);

well
S
writetable(T,'upsampled_matlab_foh2.csv')
