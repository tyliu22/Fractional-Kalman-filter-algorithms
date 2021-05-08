%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Fractional-Kalman-filter-algorithms 
%        fractional order PF
%   Purpose: performance analysis of PF
%            evaluate system noise mean
%         function:    D^{0.7} x_k = 3*sin(2*x_{k-1}) -x_{k-1} + w_k
%                              y_k = x_k + v_k
%        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;

LineWidth = 1.5;
h_con = sqrt(3);
tf = 100; % simulation interval
N = 100;  % number of particle

% system matrix
% A = [0,1; -0.1,-0.2];      % system matrix
% B = [0; 1];                %
% C = [0.1,0.3];             %
I = eye(1,1);                %
%I(3,3) = 0;

% noise
q = 0;                      % system noise mean
r = 0;                      % measure noise mean
Q = 0.81;                   % system noise variance
R = 0.25;                   % measure noise variance

W_noise = sqrt(Q)*randn(1,N) + q;  % system noise
V_noise = sqrt(R)*randn(1,N) + r;  % measure noise

x = zeros(1,tf); % system real state intialization
y = zeros(1,tf); 
y(1,1) = x(1,1) + sqrt(R) * randn;

P = zeros(1,tf);        % sampling variance
P(1,1) = 2;             % sampling distribution variance intialization
xhatPart = zeros(1,tf); % state estimation

xpart = zeros(N,tf);
for i = 1 : N
    xpart(i,1) = x(1,1) + sqrt(P(1,1)) * randn; 
end
% xArr = [x];
% yArr = [];
% xhatArr = [x];
% PArr = [P];
%xhatPartArr = [xhatPart]; %

% fractional order alpha and its corresponding GL binomial coefficient 
bino_fir = zeros(1,N);      
alpha = 1;
bino_fir(1,1) = 0.7;
for i = 2:1:N
    bino_fir(1,i) = (1-(alpha+1)/(i-1))*bino_fir(1,i-1);
end

%%
% diff_X_real    differential of state at k
diff_X_real = 0;

%%
for k = 2 : tf

    diff_X_real = 3*sin(2*x(1,k-1)) -x(1,k-1) + W_noise(1,k-1);
    rema = 0;
    for i = 2:1:k
        rema = rema + bino_fir(1,i)*x(1,k+1-i);
    end
    x(1,k) = diff_X_real - rema;
    % 
    y(1,k) = x(1,k) + V_noise(1,k);  % observation at k

 %% sampling N particles
 for i = 1 : N
     % sampling N particles
     xpartminus(i) = 3*sin(2*xpart(i,k-1)) - xpart(i,k-1) + sqrt(Q) * randn;
     temp = 0;
         for j = 2 : 1 : k
            temp = temp + bino_fir(1,j)*xpart(i,k+1-j);
         end
     xpartminus(i) = xpartminus(i) - temp;
     ypart = xpartminus(i);      % observation of each particle
     vhat = y(1,k) - ypart;      % likelihood between real observation
     q(i) = (1 / sqrt(R) / sqrt(2*pi)) * exp(-vhat^2 / 2 / R);
     %likelihood or similarity of each particle
 end

 %%
% Weight normalization
qsum = sum(q);
for i = 1 : N
    q(i) = q(i) / qsum; %normalized weight q
end

%%
 % resampling
  for i = 1 : N 
      u = rand;
      qtempsum = 0; 
      for j = 1 : N
          qtempsum = qtempsum + q(j); 
          if qtempsum >= u 
              xpart(i,k) = xpartminus(j);
              break;
          else
              xpart(i,k) = xpart(i,k-1);
          end 
      end
  end
xhatPart(1,k) = mean(xpart(:,k));

%%
%
% xArr = [xArr x];   
% yArr = [yArr y];  
% % xhatArr = [xhatArr xhat]; 
% PArr = [PArr P]; 
% xhatPartArr = [xhatPartArr xhatPart];

end

%%
t = 1 : tf;
figure;
plot(t, x, 'r', t, xhatPart, 'b--','linewidth',LineWidth);
Esitimated_state = legend('Real Value','Estimated Value','Location','best');
set(Esitimated_state,'Interpreter','latex')
set(gcf,'Position',[200 200 400 300]); 
axis([0 50 -6 6]) 
axis normal
set(gca,'FontSize',10); 
xlabel('time step','FontSize',7); 
ylabel('state','FontSize',7);
set(gca,'FontName','Helvetica','FontSize',8)
%title('Fractional particle filter')
%xhatRMS = sqrt((norm(x - xhat))^2 / tf);
%xhatPartRMS = sqrt((norm(xArr - xhatPartArr))^2 / tf);


% figure;
% plot(t,abs(x-xhatPart),'b');
% title('The error of FPF')


%%
% t = 0 : tf;
% figure;
% plot(t, xArr, 'b-.', t, xhatPartArr, 'k-');
% legend('Real Value','Estimated Value');
% set(gca,'FontSize',10); 
% xlabel('time step'); 
% ylabel('state');
% title('Particle filter')
% xhatRMS = sqrt((norm(xArr - xhatArr))^2 / tf);
% xhatPartRMS = sqrt((norm(xArr - xhatPartArr))^2 / tf);
% figure;
% plot(t,abs(xArr-xhatPartArr),'b');
% title('The error of PF')



