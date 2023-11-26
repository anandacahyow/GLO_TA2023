clear;clc;close all

% ===== INPUT =====
a = -0.001286
b = 1.489
c = 1.25
% a = -0.000844
% b = 1.208
% c = 0.424
% a = -0.000791
% b = 1.166
% c = 0.917

wc = 0.44

data = [0 0
    387	424.09
639	423.8
639	401.11
472	400.11
472	432.12
621	417.22
];

% data = [0 0
%     639	423.8
% 639	401.11
% 472	400.11
% 472	432.12
% 621	417.22
% 603	437.28
% 578	373.71
% ];

% data = [0 0
%     639	401.11
% 472	400.11
% 472	432.12
% 621	417.22
% 603	437.28
% 578	373.71
% 715	390.84
% ];


a2 = -1.5e-5
b2 = 0.13
c2 = 0.87
% a2 = -1.19e-5
% b2 = 0.116
% c2 = 0.529
% a2 = -1.14e-5
% b2 = 0.113
% c2 = 0.463

wc2 = 0.05

data2 = [0 0 
    3010	284.44
4970	275.09
4424	265.6
3675	272.92
3430	266.77
4606	281.05
4474	277.04
];
% data2 = [0 0
%     4970	275.09
% 4424	265.6
% 3675	272.92
% 3430	266.77
% 4606	281.05
% 4474	277.04
% 4327	292.33
% ];
% data2 = [0 0
%     4424	265.6
% 3675	272.92
% 3430	266.77
% 4606	281.05
% 4474	277.04
% 4327	292.33
% 4865	260.97
% ];

n = 2;
max_val = 8000

glir_opt = 578
glir_opt2 = 4327

% glir_opt = 715
% glir_opt2 = 4865
% 
% glir_opt = 737
% glir_opt2 = 4962

% ===== CALC =====
glir_optim = [glir_opt glir_opt2]
y_optim = [reg(glir_opt,wc,a,b,c) reg(glir_opt2,wc2,a2,b2,c2)]


x2 = [0:max_val];
x = [0:1/n:max_val/n];

fig = figure(1);
set(fig,'defaultAxesColorOrder',[[0 0 0];[0 0 0]]);

plot(x,reg(x,wc,a,b,c),'Color',	'#0072BD')
hold on
plot(glir_opt,reg(glir_opt,wc,a,b,c),'o','MarkerFaceColor',	'#A2142F')
text(glir_opt,reg(glir_opt,wc,a,b,c)+20,['GLIR=' '(' num2str(glir_opt) ')'])
ylim([0 max(y_optim)+50])
hold on
ylabel("Liquid Flow Well OA - 11 (STB/day)")
xlabel("Gas Lift Injection Rate (MSCFD)")


f=fit(data(:,1),data(:,2),'poly2')
f1 = reg(x,wc,f.p1,f.p2,f.p3);
fun1 = @(x)(wc-1)*(f.p1*x^2+f.p2*x+f.p3)


ff=fit(data2(:,1),data2(:,2),'poly2')
f2 = reg(x2,wc2,ff.p1,ff.p2,ff.p3);
fun2 = @(x)(wc2 - 1)*(ff.p1*x^2+ff.p2*x+ff.p3)

yyaxis right
plot(x,f1,'Color','#D95319')
hold on
plot(fminunc(fun1,0),reg(fminunc(fun1,0),wc,f.p1,f.p2,f.p3),'o','MarkerFaceColor',	'#77AC30')
text(fminunc(fun1,0),reg(fminunc(fun1,0),wc,f.p1,f.p2,f.p3)-10,['GLIR=' '(' num2str(fminunc(fun1,0)) ')'])
ylim([0 max(y_optim)+50])
grid on

legend("Well OA - 11","OA - 11 GLIR Optimized","Well OA - 11 Standalone","OA - 11 GLIR Optimized Standalone",'Location','southeast')
title("Data-Driven Gas Lift Performance Curve Well OA - 11")
xlabel("Gas Lift Injection Rate (MSCFD)")


fig2 = figure(2);
set(fig2,'defaultAxesColorOrder',[[0 0 0];[0 0 0]]);

plot(x2,reg(x2,wc2,a2,b2,c2),'Color',	'#0072BD')
hold on
plot(glir_opt2,reg(glir_opt2,wc2,a2,b2,c2),'o','MarkerFaceColor',	'#A2142F')
text(glir_opt2,reg(glir_opt2,wc2,a2,b2,c2)+20,['GLIR=' '(' num2str(glir_opt2) ')'])
ylim([0 max(y_optim)+50])
hold on
ylabel("Liquid Flow Well OA - 12 (STB/day)")

yyaxis right
plot(x2,f2,'Color',	'#D95319')
hold on
plot(fminunc(fun2,0),reg(fminunc(fun2,0),wc2,ff.p1,ff.p2,ff.p3),'o','MarkerFaceColor',	'#77AC30')
text(fminunc(fun2,0),reg(glir_opt2,wc2,ff.p1,ff.p2,ff.p3)-20,['GLIR=' '(' num2str(fminunc(fun2,0)) ')'])
%ylim([0 max(y_optim)+50])
ylim([0 350])
title("Data-Driven Gas Lift Performance Curve Well OA - 12")
legend("Well OA - 12","OA - 12 GLIR Optimized","Well OA - 12 Standalone","OA - 12 GLIR Optimized Standalone",'Location','southeast')
xlabel("Gas Lift Injection Rate (MSCFD)")
grid on

function [y] = reg(x,wc,a,b,c)
y = (1-wc)*(a*x.^2+b*x+c);
end