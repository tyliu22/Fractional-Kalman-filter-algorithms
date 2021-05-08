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

SimuTimes = 100;    % simulation interval

NumParticle = 100;  % number of particle

I = eye(1,1);               

% noise
q = 0;                      % system noise mean
r = 0;                      % measure noise mean
Q = 0.81;                   % system noise variance
R = 0.25;                   % measure noise variance

W_noise = sqrt(Q)*randn(1,SimuTimes) + q;    % system noise
V_noise = sqrt(R)*randn(1,SimuTimes) + r;    % measure noise

X_RealState = zeros(1,SimuTimes); % system real state intialization
Y_RealMeas = zeros(1,SimuTimes);  %
Y_RealMeas(1,1) = X_RealState(1,1) + sqrt(R) * randn;

P_SampleCov = zeros(1,SimuTimes);        % sampling variance
x_EstiState = zeros(1,SimuTimes);        %  state estimation
P_SampleCov(1,1) = 2;                    % sampling distribution variance intialization

ParticleWeight    = zeros(1,NumParticle);     % weight normalization
x_SamplePart_temp = zeros(1,NumParticle);     % 
x_SampleParticle  = zeros(NumParticle,SimuTimes);

% Intinialization particle, prior distirbution p(x_0) 
for i = 1 : NumParticle
    x_SampleParticle(i,1) = x_EstiState(1,1) + q + sqrt(P_SampleCov(1,1)) * randn; 
end
% xArr = [x];
% yArr = [];
% xhatArr = [x];
% PArr = [P];
% xhatPartArr = [xhatPart]; %

f = @(x)3*sin(2*x) - x;
h = @(x)x;

% fractional order alpha and its corresponding GL binomial coefficient 
bino_fir = zeros(1,SimuTimes);       % alpha = 0.7
alpha = 0.7;
bino_fir(1,1) = 1;
for i = 2:1:NumParticle
    bino_fir(1,i) = (1-(alpha+1)/(i-1))*bino_fir(1,i-1);
end

%%
% diff_X_real
diff_X_real = 0;

%% calculate real state and measurement
for k = 2 : SimuTimes
    diff_X_real = f(X_RealState(1,k-1)) + W_noise(1,k-1);
    rema = 0;
    for i = 2:1:k
        rema = rema + bino_fir(1,i) * X_RealState(1,k+1-i);
    end
    X_RealState(1,k) = diff_X_real - rema;
    % observation at k
    Y_RealMeas(1,k) = h(X_RealState(1,k)) + V_noise(1,k); 

%% sample N particles
 for i = 1 : NumParticle
     % Draw particle: x^i_k ~ p(x_k | x^i_k-1) state transform function
     % sample Num_particle particles
     x_SamplePart_temp(1,i) =  f(x_SampleParticle(i,k-1)) + q + sqrt(Q) * randn;
     temp = 0;
         for j = 2 : 1 : k
            temp = temp + bino_fir(1,j)*x_SampleParticle(i,k+1-j);
         end
     x_SamplePart_temp(1,i) = x_SamplePart_temp(1,i) - temp;
     y_ParticleMeas = h(x_SamplePart_temp(1,i)) + r;     % observation of each particle
     ErrorMeas = Y_RealMeas(1,k) - y_ParticleMeas;   % likelihood between real observation
     % Draw weight: w^i_k ~ p(z_k | x^i_k) measurement transform function
     %ParticleWeight(1,i) = h(x_SamplePart_temp(1,i)) + r + sqrt(R) * randn;
     ParticleWeight(1,i) = (1 / sqrt(R) / sqrt(2*pi)) * exp(-ErrorMeas^2 / 2 / R);
     % likelihood or similarity of each particle
 end

%%
 % Weight normalization
weight_sum = sum(ParticleWeight);
for i = 1 : NumParticle
    ParticleWeight(1,i) = ParticleWeight(1,i) / weight_sum;  % normalized weight q
end

 % resampling
 qtempsum = zeros(1,NumParticle);
  qtempsum(1,1) = ParticleWeight(1,1);
 for i = 2 : 1 : NumParticle
    qtempsum(1,i) = qtempsum(1,i-1) + ParticleWeight(1,i);
 end
 
  for i = 1 : NumParticle
      UniRandom = rand; %
      for j = 1 : NumParticle
          % 
          %qtempsum = qtempsum + ParticleWeight(1,j);
          if qtempsum(1,j) >= UniRandom
              x_SampleParticle(i,k) = x_SamplePart_temp(1,j);
              break;
          %else
          %    x_SampleParticle(i,k) = x_SampleParticle(i,k-1);
          end
      end
  end

%%
%  c_Weight = zeros(1,NumParticle);
%  c_Weight(1,1) = ParticleWeight(1,1);
% for i = 2 : 1 : NumParticle
%     c_Weight(1,i) = c_Weight(1,i-1) + ParticleWeight(1,i);
% end
%  
% i = 1;
% UniRandom = 1/NumParticle * rand; %
% 
% for j = 1 : 1 : NumParticle
%     UniRandom_j = UniRandom + 1/NumParticle * (j-1);
% 
%     if UniRandom_j > c_Weight(1,i)
%         i = i + 1;
%     end
%     x_SampleParticle(j,k) = x_SamplePart_temp(1,i);
% 
% end


%%
%   for i = 1 : NumParticle
%       UniRandom = rand; 
%       qtempsum = 0;
%       for j = 1 : NumParticle
%           qtempsum = qtempsum + ParticleWeight(1,j);
%           if qtempsum >= UniRandom
%               x_SampleParticle(i,k) = x_SamplePart_temp(1,j);
%               break;
%           %else
%           %    x_SampleParticle(i,k) = x_SampleParticle(i,k-1);
%           end
%       end
%   end
 
  
%% state estimation
  
x_EstiState(1,k) = mean(x_SampleParticle(:,k));

%%
% xArr = [xArr x];   
% yArr = [yArr y];  
% % xhatArr = [xhatArr xhat]; 
% PArr = [PArr P]; 
% xhatPartArr = [xhatPartArr xhatPart];

end

%%
t = 1 : SimuTimes;
figure;
plot(t, X_RealState, 'r', t, x_EstiState, 'b--','linewidth',LineWidth);
Esitimated_state = legend('Real Value','Estimated Value','Location','best');
set(Esitimated_state,'Interpreter','latex')
set(gcf,'Position',[200 200 400 300]); 
axis([0 50 -6 6])
axis normal
set(gca,'FontSize',10); 
xlabel('time step','FontSize',7); 
ylabel('state','FontSize',7);
set(gca,'FontName','Helvetica','FontSize',8)
% title('Fractional particle filter')
% xhatRMS = sqrt((norm(x - xhat))^2 / tf);
% xhatPartRMS = sqrt((norm(xArr - xhatPartArr))^2 / tf);

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



