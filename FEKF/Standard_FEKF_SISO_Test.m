
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Fractional-Kalman-filter-algorithms
%   Paper：     fractional order EKF
%   Purpose：performance analysis between FCDKF and FEKF
%   Example fucntion:    D^{0.7} x_k = 3*sin(2*x_{k-1}) -x_{k-1} + w_k
%                                y_k = x_k + v_k
%   Result：
%
%   Remark：
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear

%仿真步长
N = 50;

q = 1;                   % system noise mean
r = 1;                   % measure noise mean
Q = 0.81;                % system noise variance
R = 0.25;                % measure noise variance

% short memory length under GL fractional definition
L = N+1;

% fractional order alpha and its corresponding GL binomial coefficient 
bino_fir = zeros(1,N);       % Differential order 0.7
alpha = 0.7;
bino_fir(1,1) = 1;
for i = 2:1:N
    bino_fir(1,i) = (1-(alpha+1)/(i-1))*bino_fir(1,i-1);  
end

I = eye(1,1);


% State initialization 
X_real = zeros(1,N);         % real state
Z_meas = zeros(1,N);         % real measurement 

W_noise = sqrt(Q)*randn(1,N) + q;  % system noise
V_noise = sqrt(R)*randn(1,N) + r;  % measusre noise

x_0  = 0;                    % state initialization     
X_real(1,1) = x_0;           % real state intialization
Z_meas(1,1) = V_noise(1,1);  % measurement state intialization

% system function and measurement function
f=@(x)3*sin(2*x)-x;
h=@(x)x;

for k=2:1:N
    % calculate real state
    diff_X_real = f(X_real(1,k-1)) + W_noise(1,k-1);
    rema = 0;
    for i = 2:1:k
        rema = rema + bino_fir(1,i)*X_real(1,k+1-i);
    end
    X_real(1,k) = diff_X_real - rema;
    % calculate real observation
    Z_meas(1,k) = h(X_real(1,k)) + V_noise(1,k); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------ Fractional-Kalman-filter-algorithms performance testing ----------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_esti = zeros(1,N);       % state optimal estimation 
P_xesti = zeros(1,N);      % estiamtion error variance

P_pred_0 = 100;              % prediction variance intialization
P_xesti(1,1) = P_pred_0;     % estimation variance intialization

for k=2:1:N
      % state prediction :X_pre
        diff_X_esti = f(X_esti(1,k-1));
            % calculate remainder term
            rema = 0;
            if k>L
                for i = 2:1:L+1
                   rema = rema + bino_fir(1,i)*X_esti(1,k+1-i);
                end
            else
                for i = 2:1:k
                    rema = rema + bino_fir(1,i)*X_esti(1,k+1-i);
                end
            end
        X_pre = diff_X_esti - rema + q;     % first step state prediction
        % prediction error covariance: P_pred
            rema_P = 0;
            if k>L+1
                for i = 3:1:L+2
                    rema_P = rema_P + bino_fir(1,i)*P_xesti(1,k+1-i)*bino_fir(1,i)';
                end
            else
                for i = 3:1:k
                    rema_P = rema_P + bino_fir(1,i)*P_xesti(1,k+1-i)*bino_fir(1,i)';
                end
            end
        F = 6*cos(2*X_esti(1,k-1)) - 1;
            
        P_xpred = (F-bino_fir(1,2))*P_xesti(1,k-1)*(F-bino_fir(1,2))'+ Q + rema_P;
        
        % measurement estimation  Z_esti ---- Z_k|k-1
        Z_esti = h(X_pre) + r;
        
        % kalman gain: Kk(2*1)
        H = 1;
        Kk = P_xpred*H'/(H*P_xpred*H' + R);
        
        % state updating
        X_esti(1,k) = X_pre + Kk*( Z_meas(1,k) - Z_esti );
        
        % estimation variance updating
        P_xesti(1,k) = (I-Kk*H)*P_xpred;
end

k = 1:1:N;

LineWidth = 1.5;

figure;
plot(k,X_real(1,:),'r',k,X_esti(1,:),'b--','linewidth',LineWidth);
set(gcf,'Position',[200 200 400 300]); 
% axis([xmin xmax ymin ymax]) 
axis normal
axis([0 N -6 6 ])
ylabel('x','FontSize',8)
xlabel('time(sec)','FontSize',8)
set(gca,'FontName','Helvetica','FontSize',8)
legend('real state','estimated state','Location','best');





