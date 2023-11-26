clear
clc
close all

files = dir('*.csv')
TOTAL = struct;
c = []

for i = 1:length(files)
    names = files(i).name
    
    Data = importdata(names)
    %TOTAL(i).structure = Data
    TOTAL(i).Current_Time = Data.data(:,1)
    TOTAL(i).Current = Data.data(:,2)
    x = TOTAL(i).Current_Time
    y = TOTAL(i).Current
    
    %figure(i)
    %plot(x,y)
end