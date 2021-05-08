%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Fractional-Kalman-filter-algorithms
%   Paper：     fractional order CDKF
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

%Interval step 
N = 50;

h_con = sqrt(4);

q = 0.5;                  % system noise mean
r = 0.5;                  % measure noise mean
Q = 0.81;                 % system noise variance
R = 0.81;                 % measure noise variance

% short memory length under GL fractional definition
L = N+1;

%fractional order alpha and its corresponding GL binomial coefficient 
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

% Noises 
W_noise = sqrt(Q)*randn(1,N) + q;  % system noise
V_noise = sqrt(R)*randn(1,N) + r;  % measusre noise

x_0  = 0;                    %  state initialization     
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

X_esti = zeros(1,N);         % state optimal estimation 
P_xesti = zeros(1,N);        % estiamtion error variance

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
        % prediction error covariance:P_pred

            %cholsky decomposition
            S_chol = chol(P_xesti(1,k-1))';

            % calculate remainder term
            rema_P = 0;
            if k>L+1
                for i = 2:1:L+2
                    rema_P = rema_P + bino_fir(1,i)*P_xesti(1,k+1-i)*bino_fir(1,i)';
                end
            else
                for i = 2:1:k
                    rema_P = rema_P + bino_fir(1,i)*P_xesti(1,k+1-i)*bino_fir(1,i)';
                end
            end

        % temporal variable temp_fun
        temp_fun = f(X_esti(1,k-1)+h_con*S_chol) - f(X_esti(1,k-1)-h_con*S_chol);
        temp = 1/(4*h_con^2) * temp_fun^2 + rema_P + ...
                  1/h_con*0.5*temp_fun*S_chol'*(-bino_fir(1,2))' + ...
                  1/h_con*(-bino_fir(1,2))*S_chol*0.5*temp_fun';
        P_xpred = temp + Q;
        
        % measurement estimation  Z_esti ---- Z_k|k-1
        Z_esti = h(X_pre) + r;

        % measurement error covariance: P_zpred ---- P_z_k|k-1
        P_zpred = P_xpred + R;

        % kalman gain: Kk(2*1)
        Kk = P_xpred/P_zpred;

        % state updating
        X_esti(1,k) = X_pre + Kk*( Z_meas(1,k) - Z_esti );

        % estimation variance updating
        P_xesti(1,k) = P_zpred - Kk*P_zpred*Kk';
end

% plot
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







