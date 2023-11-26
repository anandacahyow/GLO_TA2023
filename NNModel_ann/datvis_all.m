clc;clf

well1= readtable("oa-11.xlsx")
well2 = readtable("oa-12.xlsx")

well1 = fillmissing(well1,'constant',0,'DataVariables',@isnumeric)
well2 = fillmissing(well2,'constant',0,'DataVariables',@isnumeric)

% figure(1)
% stackedplot(well1(:,[7 8 11 14 16]),'o-')
% grid on
% title("Well OA-11")
% xlabel("Day(s)")
% 
% figure(2)
% stackedplot(well2(:,[8 9 12 15 17]),'o-')
% grid on
% title("Well OA-12")
% xlabel("Day(s)")


figure(3)
plot(well2.START_TEST(32:end),well2.CASING_A(32:end),'-o','MarkerFaceColor','#0072BD')
ylabel("Casing Head Pressure (psia)")
hold on
yyaxis right
plot(well2.START_TEST(32:end),well2.GAS_LIFT_RATE(32:end),'-o','MarkerFaceColor','#D95319')
ylabel("Gas Lift Injection Rate (MMSCFD)")
legend("Casing Head Pressure (CHP)","Gas Lift Injection Rate (GLIR)")
grid on
title("Well OA - 12")

figure(4)
plot(well1.START_TEST(32:end-1),well1.CASING_A(32:end-1),'-o','MarkerFaceColor','#0072BD')
ylabel("Casing Head Pressure (psia)")
hold on
yyaxis right
plot(well1.START_TEST(32:end-1),well1.GAS_LIFT_RATE(32:end-1),'-o','MarkerFaceColor','#D95319')
ylabel("Gas Lift Injection Rate (MMSCFD)")
legend("Casing Head Pressure (CHP)","Gas Lift Injection Rate (GLIR)")
grid on
title("Well OA - 11")

% figure(3)
% stackedplot(well1)
% grid on
% title("Well OA-11")

%summary(well1)
%summary(well2)
%head(well1)
%head(well2)