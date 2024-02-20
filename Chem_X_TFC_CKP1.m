%%
clear; close all; clc;
format longE
%--------------------------------------------------------------------------
%{
  Physics-Informed X-TFC applied to Stiff Chemical Kinetics


  Authors:
  Mario De Florio - PhD Candidate, The University of Arizona
  Enrico Schiassi - PhD Candidate, The University of Arizona
  Matthew Frankel - PhD Candidate, The University of Texas at Austin  
%}
%--------------------------------------------------------------------------
%% Input Paramaters

%Set seed for reproducability
seed=575; 
rng(seed);

%Record start time
tic;

%Define Paramaters

t_0 = 1e-5; % initial time (seconds)
t_f = 3600*24*7; % final time (seconds)

n_x = 20;    % Discretization order for x (-1,1)
L = 20;    % number of neurons

%Define domain for the scaled time variable
x_min=0;
x_max=1;

%Create vector for scaled time
x1 = linspace(x_min,x_max,n_x)';

n_t = 200; %Number of time sub-domains

%Times to be used at the boundaries of each subinterval, logspaced between
%t_0 and t_f
t_tot = logspace(log10(t_0),log10(t_f),n_t)'; 

%Build a vector with all training time points for all time sub-domains
%based on linear spacing between the points of t_tot

t_all=[];
len_t_all=[];
for i= 1:(n_t-1)
    t=linspace(t_tot(i),t_tot(i+1),n_x);
    if i~=n_t-1
        t_all=[t_all t(1:end-1)];
    end

    if i==n_t-1
        t_all=[t_all t];
    end
    len_t_all(i)=length(t_all);
end

LB = -1; % Lower boundary for weight and bias samplings
UB = 1; % Upper boundary for weight and bias samplings

% iterative least-square parameters
IterMax = 300; %Max number of iterations
IterTol = 1e-12; %Tolerance to be used for stopping iterating for convergence
 
%Select range of pH values to use for training
ph_range = 7:1:10; 

%Extract the number of different pH values
n_pH=length(ph_range); 

%% Evaluate Activation Function

%Create a grid of the values of time and pH to be used for training and
%reshape variables
[X1,X2]=meshgrid(x1,ph_range);
x1_all=reshape(X1',[],1);
pH=reshape(X2',[],1);
x=[x1_all pH]; 

%Convert the domain of x2_range to -1,1
%Note: the variable x1 for time is already scaled between 0 and 1, assigned
%as such
x2_range_domain=-1+1/((max(ph_range)-min(ph_range))/(1--1))*(ph_range-min(ph_range));
[X1,X2]=meshgrid(x1,x2_range_domain);
x1_all=reshape(X1',[],1);
x2_domain=reshape(X2',[],1);
%x_act is an array of x_1 and x_2 values which will be used as inputs to
%the activation functionn
x_act = [x1_all x2_domain];


%Uniform random initiation of weights and biases
%Uses the same weights and biases for each discretized model
%Weights and biases remain constant
weight = unifrnd(LB,UB,L,2);
bias = unifrnd(LB,UB,L,1);

%Initialize variables for sigma computations
h= zeros(n_x,L); %sigma
hd= zeros(n_x,L); %first derivative of sigma

%Compute the values of each of the activation functions for each
%weight/bias combo

%Make a copy of x in which the times are always 0 to be evaluated in sigma,
%used in constrained expression
x_t_0=x_act;
x_t_0(:,1)=x_min;

%For each value of time and pH
for i = 1 : n_x*n_pH
    %For each neuron
    for j = 1 : (L)
        %Evaluate the activation function
        [h(i, j), hd(i, j)] = tanh_act(x_act(i,:)',weight(j,:), bias(j));
        %Evaluate the activation function with 0 as x_1, representing time
        %0
        [h_t_0(i, j), hd_t_0(i, j)] = tanh_act(x_t_0(i,:)',weight(j,:), bias(j));
    end
end


%% Initialize variables

%Initialize variables

%Species result for each time subdomain
y1 = zeros(n_t*n_pH,1);
y2 = zeros(n_t*n_pH,1);
y3 = zeros(n_t*n_pH,1);
y4 = zeros(n_t*n_pH,1);

%Concattenated species results for all time subdomains
y_1_all=[];
y_2_all=[];
y_3_all=[];
y_4_all=[];

%Store optimized xi values for each time subdomain
xi_1_all=[];
xi_2_all=[];
xi_3_all=[];
xi_4_all=[];


%% Define chemical paramaters and optimization hyperparamaters

% Define initial values
y1_0 = 2.142857e-05; %Initial TOTNH concentration 
y2_0 = 0;  %Initial TOTCl concentration
y3_0 = 4.22e-5; %Initial NH2Cl concentration
y4_0 = 0; %Initial NHCl2 concentration

%Save values for later for the ode solver
y1_0_initial = y1_0; 
y2_0_initial = y2_0; 
y3_0_initial = y3_0;
y4_0_initial = y4_0;

%Fill in initial values
y1(1) = y1_0;
y2(1) = y2_0;
y3(1) = y3_0;
y4(1) = y4_0;

%Create vector to be filled in later
training_err_vec = zeros(n_t-1,1);

%Calculate the values of alpha to be used based on the values of pH
al0_cl=1./(1+(3.16*10^-8./(10.^-pH))); %alpha_0 value for HOCl/OCl- system
al1_nh=1./(1+((10.^-pH)./(5*10^-10))); %alpha_1 for NH4+/NH3 system

%Define the weights for each loss function
L1_weight=1;
L2_weight=3;
L3_weight=1;
L4_weight=1;

%Define the learning rate
LR=.25;

% Define reaction rate constants
k1 =  4.2e6 ; 
k2 =  2.1e-5; 
k3 =  280 ;

%% Train X-TFC for each time subdomain 
for i = 1:(n_t-1)

    %Update the value of c_i based on which times of the sub-interval 
    c_i = (x_max - x_min) / (t_tot(i+1) - t_tot(i));
    
    %Initialize Beta values
    xi_1_i = zeros(L,1);
    xi_2_i = zeros(L,1);
    xi_3_i = zeros(L,1);
    xi_4_i = zeros(L,1);

    %Concatenate beta values into one vector
    xi_i = [xi_1_i;xi_2_i;xi_3_i;xi_4_i];


    %% Build Constrained Expressions

    %Define constrained expressions
    y1_i = (h-h_t_0)*xi_1_i + y1_0;
    y2_i = (h-h_t_0)*xi_2_i + y2_0;
    y3_i = (h-h_t_0)*xi_3_i + y3_0;
    y4_i = (h-h_t_0)*xi_4_i + y4_0;
    
    %Define first derivative of constrained expressions w.r.t. time
    y1_dot_i = c_i*hd*xi_1_i;
    y2_dot_i = c_i*hd*xi_2_i;
    y3_dot_i = c_i*hd*xi_3_i;
    y4_dot_i = c_i*hd*xi_4_i; 
    
    %% Build the Losses

    %Define reaction terms to be used in loss functions
    r1=k1.*al0_cl.*y2_i.*al1_nh.*y1_i;
    r2=k2.*y3_i;
    r3=k3.*al0_cl.*y2_i.*y3_i;

    %Define loss terms
    L_1 = L1_weight.*(-r1 + r2 -y1_dot_i) ;
    L_2 = L2_weight.*(-r1 + r2 - r3 - y2_dot_i) ;
    L_3 = L3_weight.*(r1 - r2 - r3 - y3_dot_i) ;
    L_4 = L4_weight.*(r3 - y4_dot_i) ;
    
    %Concattenate loss terms into one vector
    Loss = [L_1 ; L_2 ; L_3 ;L_4];

    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;%Counter for number of training iterations


    %Define some variables that are used often in following calculations
    %Derivative of y wrt its own beta (dy1/dx1, dy2/dx2, etc...)
    dydx = h-h_t_0;

    %Derivative of y' wrt its own beta (dy'1/dx1, dy'2/dx2, etc...)
    dy_primedx = c_i .* hd;

    %% Tune beta values using iterative least squares
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        %Extract l2 norm of loss vector from previous iteration
        l2(1)= l2(2);

        % Compute partial derivatives used to construct jacobian matrix
        
        %Derivative of L1 with respect to each beta value
        L_y1_xi_1 = L1_weight.*(-k1.*al0_cl.*y2_i.*al1_nh.*dydx - dy_primedx); 
        L_y1_xi_2 = L1_weight.*(-k1.*al0_cl.*y1_i.*al1_nh.*dydx);
        L_y1_xi_3 = L1_weight.*(k2.*dydx);
        L_y1_xi_4 = L1_weight.*(zeros(n_x*n_pH,L)) ;

        %Derivative of L2 with respect to each beta value
        L_y2_xi_1 = L2_weight.*(-k1.*al0_cl.*y2_i.*al1_nh.*dydx );
        L_y2_xi_2 = L2_weight.*(-k1.*al0_cl.*y1_i.*al1_nh.*dydx - k3.*al0_cl.*y3_i.*dydx -dy_primedx);
        L_y2_xi_3 = L2_weight.*(k2.*dydx - k3.*al0_cl.*y2_i.*dydx) ;        
        L_y2_xi_4 = L2_weight.*(zeros(n_x*n_pH,L) );

        %Derivative of L3 with respect to each beta value
        L_y3_xi_1 = L3_weight.*(k1.*al0_cl.*y2_i.*al1_nh.*dydx ); 
        L_y3_xi_2 = L3_weight.*(k1.*al0_cl.*y1_i.*al1_nh.*dydx - k3.*al0_cl.*y3_i.*dydx); 
        L_y3_xi_3 = L3_weight.*(-k2.*dydx - k3*al0_cl.*y2_i.*dydx -dy_primedx);        
        L_y3_xi_4 = L3_weight.*(zeros(n_x*n_pH,L) );

        %Derivative of L4 with respect to each beta value
        L_y4_xi_1 = L4_weight.*(zeros(n_x*n_pH,L)); 
        L_y4_xi_2 = L4_weight.*(k3.*al0_cl.*y3_i.*dydx) ; 
        L_y4_xi_3 = L4_weight.*(k3.*al0_cl.*y2_i.*dydx);        
        L_y4_xi_4 = L4_weight.*(-dy_primedx) ;


        % Build the jacobian matrix
        JJ = [ L_y1_xi_1 , L_y1_xi_2 , L_y1_xi_3, L_y1_xi_4 ; 
               L_y2_xi_1 , L_y2_xi_2 , L_y2_xi_3, L_y2_xi_4 ;
               L_y3_xi_1 , L_y3_xi_2 , L_y3_xi_3, L_y3_xi_4 ;
               L_y4_xi_1 , L_y4_xi_2 , L_y4_xi_3, L_y4_xi_4 ];  

        % Determine the gradient of the beta values
        dxi = lsqminnorm(JJ,Loss);

        % update beta values based on gradient and 
        xi_i = xi_i - LR.*dxi;

        xi_1_i = xi_i(1:L);
        xi_2_i = xi_i(L+1:2*L);
        xi_3_i = xi_i((2*L)+1:3*L);
        xi_4_i = xi_i((3*L)+1:4*L);


        %% Re-Build Constrained Expressions
        y1_i = (h-h_t_0)*xi_1_i + y1_0;          
        y2_i = (h-h_t_0)*xi_2_i + y2_0;           
        y3_i = (h-h_t_0)*xi_3_i + y3_0;            
        y4_i = (h-h_t_0)*xi_4_i + y4_0;          

        y1_dot_i = c_i*hd*xi_1_i;
        y2_dot_i = c_i*hd*xi_2_i;
        y3_dot_i = c_i*hd*xi_3_i;
        y4_dot_i = c_i*hd*xi_4_i;
        %% Re-Build the Losses
        r1=k1.*al0_cl.*y2_i.*al1_nh.*y1_i;
        r2=k2.*y3_i;
        r3=k3.*al0_cl.*y2_i.*y3_i;

        L_1 = L1_weight.*(-r1+ r2  -y1_dot_i) ;
        L_2 = L2_weight.*(-r1+ r2- r3  -y2_dot_i) ;
        L_3 = L3_weight.*(r1 - r2 - r3 -y3_dot_i) ;
        L_4 = L4_weight.*(r3 -y4_dot_i) ;

        Loss = [L_1 ; L_2 ; L_3 ;L_4]; 

        l2(2) = norm(Loss);


        %Derivative of y wrt its own beta (dy1dx1, dy2dx2, etc...)
        dydx = h-h_t_0;

        %Derivative of y' wrt its own beta (dy'1dx1, dy'2dx2, etc...)
        dy_primedx = c_i .* hd;

        iter = iter+1;
    end

    %% Extract beta values and initial conditions for subsequent time sub-domain

    %Fill in the xi values with the trained values of
    %Beta (xi) so that they can be used later when evaluating the model for
    %out-of-sample pH values
   
    xi_1_all=[xi_1_all xi_1_i];
    xi_2_all=[xi_2_all xi_2_i];
    xi_3_all=[xi_3_all xi_3_i];
    xi_4_all=[xi_4_all xi_4_i];

    %Assign the final value of each species for a time sub-interval as the
    %initial value for the following sub-interval
    
    %Initial value of y1
    y1_i_mat=reshape(y1_i,n_x,n_pH);
    y1_0 = y1_i_mat(end,:)';
    r=repmat(y1_0',n_x,1) ;
    y1_0=r(:) ;

    %Initial value of y2
    y2_i_mat=reshape(y2_i,n_x,n_pH);
    y2_0 = y2_i_mat(end,:)';
    r=repmat(y2_0',n_x,1) ;
    y2_0=r(:) ;

    %Initial value of y3
    y3_i_mat=reshape(y3_i,n_x,n_pH);
    y3_0 = y3_i_mat(end,:)';
    r=repmat(y3_0',n_x,1) ;
    y3_0=r(:) ;

    %Initial value of y4
    y4_i_mat=reshape(y4_i,n_x,n_pH);
    y4_0 = y4_i_mat(end,:)';
    r=repmat(y4_0',n_x,1) ;
    y4_0=r(:) ;


    %Save the results of the model in each of the training points
    if i ~=(n_t-1)
        %y1
        z=reshape(y1_i,n_x,n_pH);
        y_1_all=[y_1_all; z(1:end-1,:)];

        %y2
        z=reshape(y2_i,n_x,n_pH);
        y_2_all=[y_2_all; z(1:end-1,:)];

        %y3
        z=reshape(y3_i,n_x,n_pH);
        y_3_all=[y_3_all; z(1:end-1,:)];

        %y4
        z=reshape(y4_i,n_x,n_pH);
        y_4_all=[y_4_all; z(1:end-1,:)];

    end

    if i ==(n_t-1)
     y_1_all=[y_1_all; reshape(y1_i,n_x,n_pH)];
     y_2_all=[y_2_all; reshape(y2_i,n_x,n_pH)];
     y_3_all=[y_3_all; reshape(y3_i,n_x,n_pH)];
     y_4_all=[y_4_all; reshape(y4_i,n_x,n_pH)];

    end   

%End of model training for all time sub-intervals
end 

toc

%% Plot Training Results

%Define colors of plots for in-sample pH values
colors=[191, 87, 0;
        255 214 0
        87, 157, 66;
        0, 95, 134;
        ]/255;

%Create figure
figure

%Plot X-TFC Results
for k=1:length(ph_range)
    subplot(9,2,[1 3 5 7])
    plot(t_all/3600/24,y_1_all(:,k),'color',colors(k,:),'LineWidth',1.5)
    hold on

    subplot(9,2,[2 4 6 8])
    plot(t_all/3600/24,y_2_all(:,k),'color',colors(k,:),'LineWidth',1.5)
    hold on  

    subplot(9,2,[9 11 13 15])
    plot(t_all/3600/24,y_3_all(:,k),'color',colors(k,:),'LineWidth',1.5)
    hold on

    subplot(9,2,[10 12 14 16])
    plot(t_all/3600/24,y_4_all(:,k),'color',colors(k,:),'LineWidth',1.5)
    hold on

end

%Plot Chemical Model Results

%Options for solver
opts = odeset('RelTol',1e-12,'AbsTol',1e-12);

for i=1:length(ph_range)
    
    %Solve using ode15s for a specific pH value
    [t,y] = ode15s(@(t,y) myODE(t,y,ph_range(i)), [t_0 t_f], [y1_0_initial y2_0_initial y3_0_initial y4_0_initial],opts);

    %Plot result
    subplot(9,2,[1 3 5 7])
    plot(t/3600/24,y(:,1),'--','color',colors(i,:)/1.4,'LineWidth',2)
    hold on

    subplot(9,2,[2 4 6 8])
    hold on
    plot(t/3600/24,y(:,2),'--','color',colors(i,:)/1.4,'LineWidth',2)

    subplot(9,2,[9 11 13 15])
    hold on
    plot(t/3600/24,y(:,3),'--','color',colors(i,:)/1.4,'LineWidth',2)

    subplot(9,2,[10 12 14 16])
    hold on
    plot(t/3600/24,y(:,4),'--','color',colors(i,:)/1.4,'LineWidth',2)

end

%Add labels, limits, legend, adjust plot sizes

font_size=8;

subplot(9,2,[1 3 5 7])
hAx=gca;
hAx.Position=hAx.Position.*[1 1 1 0.85];
xlim([0 max(t_all/3600/24)])
ylabel('Concnetration (mol/l)','FontWeight','bold')

y_lim=ylim;
x_lim=xlim;
text((x_lim(2)-x_lim(1))*.03+x_lim(1), (y_lim(2)-y_lim(1))*.93+y_lim(1),'a) TOTNH','fontsize',font_size)

subplot(9,2,[2 4 6 8])
hAx=gca;
hAx.Position=hAx.Position.*[1 1 1 0.85];
xlim([0 max(t_all/3600/24)])
ylim([0 4.5*10^-9])
y_lim=ylim;
x_lim=xlim;
text((x_lim(2)-x_lim(1))*.03+x_lim(1), (y_lim(2)-y_lim(1))*.93+y_lim(1),'b) TOTCl','fontsize',font_size)

subplot(9,2,[9 11 13 15])
hAx=gca;
hAx.Position=hAx.Position.*[1 1 1 0.85];
xlabel('Time (Days)','FontWeight','bold')
xlim([0 max(t_all/3600/24)])
ylabel('Concnetration (mol/l)','FontWeight','bold')
y_lim=ylim;
x_lim=xlim;
text((x_lim(2)-x_lim(1))*.03+x_lim(1), (y_lim(2)-y_lim(1))*.1+y_lim(1),'c) NH_2Cl','fontsize',font_size)

subplot(9,2,[10 12 14 16])
hAx=gca;
hAx.Position=hAx.Position.*[1 1 1 0.85];
xlabel('Time (Days)','FontWeight','bold')
xlim([0 max(t_all/3600/24)])

y_lim=ylim;
x_lim=xlim;
text((x_lim(2)-x_lim(1))*.03+x_lim(1), (y_lim(2)-y_lim(1))*.9+y_lim(1),'d) NHCl_2','fontsize',font_size)


%Add Legend
hl=subplot(9,2,[17.5]);
%Dummy plots to show lines on the legend
plot(NaN,NaN,'color',colors(1,:),'LineWidth',1.5)
hold on
plot(NaN,NaN,'color',colors(2,:),'LineWidth',1.5)
plot(NaN,NaN,'color',colors(3,:),'LineWidth',1.5)
plot(NaN,NaN,'color',colors(4,:),'LineWidth',1.5)
plot(NaN,NaN,'k','LineWidth',1.5)
plot(NaN,NaN,'--k','LineWidth',1.5)
leg=legend('pH 7','pH 8','pH 9','pH 10','X-TFC (solid)','Chemical Model (dashed)','Orientation','horizontal');
leg.NumColumns=3;
leg.Position=[0.25,0.03,0.5,0.07];
leg.FontSize=8;
axis(hl,'off');

%% Plot out-of-sample pH results

colors_val=[248 151 31;
        166, 205, 87;
        0, 169, 183;
        ]/255;

%evaluate the mdoel based on a different value of pH not used during
%training

%Figure for validation plot
f_val=figure;

%Define pH values used for evaluation
pH=[7.5 8.5 9.5 ];
%Convert to within scaled domain
pH_in_range=-1+1/((max(ph_range)-min(ph_range))/(1--1))*(pH-min(ph_range));

%Loop through each value of pH to be evaluated
for k=1:length(pH)

    %Definie initial concentration
    y1_0 = y1_0_initial;
    y2_0 = y2_0_initial;
    y3_0 = y3_0_initial;
    y4_0 = y4_0_initial;


    %Evaluate the activation function for the desired pH value
    x=[x1 ones(length(x1),1)*pH_in_range(k)];
    x_t_0=x;
    x_t_0(:,1)=0;

    %Clear h and h_t_0 arrays from the previous evaluation
    clear h h_t_0 

    %Evaluate activation function
    for i = 1 : n_x
        %For each neuron
        for j = 1 : (L)
            %Later we will need to evaluate sigma at the actual values of x and
            %k and also at the value of time 0 and k, so doing that here
            h(i, j) = tanh_act(x(i,:)',weight(j,:), bias(j));

            %h_t_0 is the value of sigma evaluated at time of 0 and k of
            %whatever it is
            h_t_0(i, j) = tanh_act(x_t_0(i,:)',weight(j,:), bias(j));

        end
    end

    %Initialize variables
    y1_all=[];
    y1=zeros(n_t,1);
    y1(1)=y1_0;

    y2_all=[];
    y2=zeros(n_t,1);
    y2(1)=y2_0;

    y3_all=[];
    y3=zeros(n_t,1);
    y3(1)=y3_0;

    y4_all=[];
    y4=zeros(n_t,1);
    y4(1)=y4_0;


    %Evaluate constrained expressions and concatenate results
    for i = 1:(n_t-1)

        %Use the values of xi for time-subdomain
        y1_i=h*xi_1_all(:,i)-h_t_0*xi_1_all(:,i)+y1_0;
        y2_i=h*xi_2_all(:,i)-h_t_0*xi_2_all(:,i)+y2_0;
        y3_i=h*xi_3_all(:,i)-h_t_0*xi_3_all(:,i)+y3_0;
        y4_i=h*xi_4_all(:,i)-h_t_0*xi_4_all(:,i)+y4_0;

        %Concatenate results for each time sub-domain
        if i ~=(n_t-1)

            y1_all = [y1_all ; y1_i(1:end-1)];
            y2_all = [y2_all ; y2_i(1:end-1)];
            y3_all = [y3_all ; y3_i(1:end-1)];
            y4_all = [y4_all ; y4_i(1:end-1)];

        else

            y1_all = [y1_all ; y1_i];
            y2_all = [y2_all ; y2_i];
            y3_all = [y3_all ; y3_i];
            y4_all = [y4_all ; y4_i];

        end
        
        %Use final condition from sub-domain as initial condition for
        %subsequent subdomain
        y1(i+1) = y1_i(end);
        y1_0 = y1_i(end);

        y2(i+1) = y2_i(end);
        y2_0 = y2_i(end);

        y3(i+1) = y3_i(end);
        y3_0 = y3_i(end);

        y4(i+1) = y4_i(end);
        y4_0 = y4_i(end);
    end

    %Plot Results
    subplot(9,2,[1 3 5 7])
    plot(t_tot/3600/24,y1,'color',colors_val(k,:),'linewidth',1.5)
    hold on

    subplot(9,2,[2 4 6 8])
    plot(t_tot/3600/24,y2,'color',colors_val(k,:),'linewidth',1.5)
    hold on

    subplot(9,2,[9 11 13 15])
    plot(t_tot/3600/24,y3,'color',colors_val(k,:),'linewidth',1.5)
    hold on

    subplot(9,2,[10 12 14 16])
    plot(t_tot/3600/24,y4,'color',colors_val(k,:),'linewidth',1.5)
    hold on

    [t,y] = ode15s(@(t,y) myODE(t,y,pH(k)), [t_0 t_f], [y1_0_initial y2_0_initial y3_0_initial y4_0_initial],opts);

    subplot(9,2,[1 3 5 7])
    plot(t/3600/24,y(:,1),'--','color',colors_val(k,:)/1.4,'LineWidth',2)
    hold on

    subplot(9,2,[2 4 6 8])
    hold on
    plot(t/3600/24,y(:,2),'--','color',colors_val(k,:)/1.4,'LineWidth',2)


    subplot(9,2,[9 11 13 15])
    hold on
    plot(t/3600/24,y(:,3),'--','color',colors_val(k,:)/1.4,'LineWidth',2)


    subplot(9,2,[10 12 14 16])
    hold on
    plot(t/3600/24,y(:,4),'--','color',colors_val(k,:)/1.4,'LineWidth',2)
end

%Adjust axis sizes and add labels, legend, etc.
subplot(9,2,[1 3 5 7])
%title('Y1 TOTNH')
ylabel('Concentration (mol/l)','FontWeight','bold')
hAx=gca;
hAx.Position=hAx.Position.*[1 1 1 0.85];
xlim([0 max(t/3600/24)])

y_lim=ylim;
x_lim=xlim;
text((x_lim(2)-x_lim(1))*.03+x_lim(1), (y_lim(2)-y_lim(1))*.91+y_lim(1),'a) TOTNH','fontsize',font_size)

subplot(9,2,[2 4 6 8])
%title('Y2 TOTCL')
hAx=gca;
hAx.Position=hAx.Position.*[1 1 1 0.85];
xlim([0 max(t/3600/24)])

y_lim=ylim;
x_lim=xlim;
text((x_lim(2)-x_lim(1))*.67+x_lim(1), (y_lim(2)-y_lim(1))*0.1+y_lim(1),'b) TOTCl','fontsize',font_size)

subplot(9,2,[9 11 13 15])
%title('Y3 NH2Cl')
xlabel('Time (days)','FontWeight','bold')
ylabel('Concentration (mol/l)','FontWeight','bold')
hAx=gca;
hAx.Position=hAx.Position.*[1 1 1 0.85];
xlim([0 max(t/3600/24)])

y_lim=ylim;
x_lim=xlim;
text((x_lim(2)-x_lim(1))*.03+x_lim(1), (y_lim(2)-y_lim(1))*.1+y_lim(1),'c) NH_2Cl','fontsize',font_size)

subplot(9,2,[10 12 14 16])
hAx=gca;
hAx.Position=hAx.Position.*[1 1 1 0.85];
%title('Y4 NHCl2')
xlabel('Time (days)','FontWeight','bold')
xlim([0 max(t/3600/24)])

y_lim=ylim;
x_lim=xlim;
text((x_lim(2)-x_lim(1))*.03+x_lim(1), (y_lim(2)-y_lim(1))*.91+y_lim(1),'d) NHCl_2','fontsize',font_size)

hl=subplot(9,2,[17.5]);
plot(NaN,NaN,'color',colors_val(1,:),'LineWidth',1.5)
hold on
plot(NaN,NaN,'color',colors_val(2,:),'LineWidth',1.5)
plot(NaN,NaN,'color',colors_val(3,:),'LineWidth',1.5)
plot(NaN,NaN,'k','LineWidth',1.5)
plot(NaN,NaN,'--k','LineWidth',2)
leg=legend('pH 7.5','pH 8.5','pH 9.5','X-TFC (solid)','Chemical Model (dashed)','Orientation','horizontal');
leg.NumColumns=3;
leg.Position=[0.25,0.03,0.5,0.07];
leg.FontSize=8;
axis(hl,'off');

%% Functions

%Define chemical ode system
function dydt = myODE(t,y,pH)
  
%Alpha Values
al0=1./(1+(3.16*10^-8./(10.^-pH))); %alpha0 for TOTCl to represent HOCl
al1=1./(1+((10.^-pH)./(5*10^-10))); %alpha1 for TOTNH to represent NH3

%Rate constants
k1 =  4.2e6 ;
k2 =  2.1e-5;
k3 =  280 ;  

%Reaction terms
a1=k1.*al0.*y(2).*al1.*y(1);
a2=k2.*y(3);
a3=k3.*al0.*y(2).*y(3);

%Growth/decay of each species
dy(1) = -a1 + a2;
dy(2) = -a1 + a2 - a3;
dy(3) = a1 - a2 - a3;
dy(4) = a3;
dydt=[dy(1);dy(2); dy(3); dy(4)];
end


%Sigmoid activation function
function [act, actd] = tanh_act(x,w,b)
    %Sigmoid activation
    act = (exp(b + (w*x)) - exp(- b - w*x))/(exp(b + w*x) + exp(- b - w*x));        
    
    %First derivative of sigmoid activation function w.r.t. first input
    %variable (representing time in this implementation)
    actd =(w(1)*exp(b + w*x) + w(1)*exp(- b - w*x))/(exp(b + w*x) + exp(- b - w*x)) - ((exp(b + w*x) - exp(- b - w*x))*(w(1)*exp(b + w*x) - w(1)*exp(- b - w*x)))/(exp(b + w*x) + exp(- b - w*x))^2;  
end

