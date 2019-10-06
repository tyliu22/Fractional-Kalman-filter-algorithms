function [SigmaWeight, SigmaPoints] = ukf_sample(state_dimension, x_state, P_covariance)
%UKF_SAMPLE 无迹卡尔曼滤波器 对称采样函数
%
kappa = 1;

%对称采样点
SigmaPoints(:,1) = x_state;

temp = sqrt(state_dimension+kappa)* chol(P_covariance)';

for i = 1 : 1 : state_dimension 
    SigmaPoints(:,i+1) = x_state + temp(:,i);
end

for i = 1 : 1 : state_dimension 
    SigmaPoints(:,state_dimension+1+i) = x_state - temp(:,i);
end

%权值

SigmaWeight(1,1) = kappa/(state_dimension+kappa);
for i = 2 : 1 : 2*state_dimension+1
    SigmaWeight(1,i) = 1/(2*(state_dimension+kappa));
end

end

