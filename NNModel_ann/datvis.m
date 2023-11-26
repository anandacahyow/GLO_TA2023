clear;clc;close all

% import data
well = importdata('well_corr.csv')
%readtable('well2.csv')
head(readtable('well_corr.csv'))

% data preparation
% date = well.textdata(2:end,1);
% date = datenum(date);

date = well.textdata(:,1);
glir1 = well.data(:,1);
qo1 = well.data(:,4);
qt1 = well.data(:,6);
wc1 = well.data(:,7);
ch1 = well.data(:,3);
gor1 = well.data(:,9);

glir2 = well.data(:,10);
qo2 = well.data(:,13);
qt2 = well.data(:,15);
wc2 = well.data(:,16);
ch2 = well.data(:,12);
gor2 = well.data(:,18);

t = [0:length(glir1)-1];

n_sampling = 0.03;
%n_sampling = 1/(24*60);

%% simulated
close all
glir3 = glir1*3;

plot(glir1)
hold on
%plot(glir2)
plot(glir3)
legend

%% System Out
glir11 = out.well1(:,1);
qo11 = out.well1(:,2);
qt11 = out.well1(:,3);
wc11 = out.well1(:,4);
ch11 = out.well1(:,5);
gor11 = out.well1(:,6);

glir22 = out.well2(:,1);
qo22 = out.well2(:,2);
qt22 = out.well2(:,3);
wc22 = out.well2(:,4);
ch22 = out.well2(:,5);
gor22 = out.well2(:,6);

index = [1:length(glir11)]';

T = table(index,glir11,qo11,qt11,wc11,ch11,gor11,glir22,qo22,qt22,wc22,ch22,gor22);
head(readtable('well'))
head(T)

S = table2struct(T);

well
S
%writetable(T,'upsampled_corr_2.csv')

%% Stacked Plot
close all
T1 = table(qo11,glir11,wc11,ch11,gor11);
T2 = table(qo22,glir22,wc22,ch22,gor22);

figure(1)
sp1 = stackedplot(T1,'-o')
grid on
title("Well OA-11")
xlabel("Day(s)")
set(sp1, 'DisplayLabels',["OIL (STB/Day)" "GAS_LIFT_RATE(MSCF/Day)" "WATERCUT_PCT (%)" "CASING_A (psia)" "GOR (Scf/STB)"])

figure(2)
sp2 = stackedplot(T2,'o-')
grid on
title("Well OA-12")
xlabel("Day(s)")
set(sp2, 'DisplayLabels',["OIL (STB/Day)" "GAS_LIFT_RATE(MSCF/Day)" "WATERCUT_PCT (%)" "CASING_A (psia)" "GOR (Scf/STB)"])