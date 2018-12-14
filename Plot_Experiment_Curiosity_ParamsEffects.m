function [] = Plot_Experiment_Curiosity_ParamsEffects(MDP,mdp)

% auxiliary plotting routine for spm_MDP_VB - multiple trials, curiosity
% paper
%
% MDP.P(M,T)      - probability of emitting action 1,...,M at time 1,...,T
% MDP.Q(N,T)      - an array of conditional (posterior) expectations over
%                   N hidden states and time 1,...,T
% MDP.X           - and Bayesian model averages over policies
% MDP.R           - conditional expectations over policies
% MDP.O(O,T)      - a sparse matrix encoding outcomes at time 1,...,T
% MDP.S(N,T)      - a sparse matrix encoding states at time 1,...,T
% MDP.U(M,T)      - a sparse matrix encoding action at time 1,...,T
% MDP.W(1,T)      - posterior expectations of precision
%
% MDP.un  = un    - simulated neuronal encoding of hidden states
% MDP.xn  = Xn    - simulated neuronal encoding of policies
% MDP.wn  = wn    - simulated neuronal encoding of precision
% MDP.da  = dn    - simulated dopamine responses (deconvolved)
% MDP.rt  = rt    - simulated dopamine responses (deconvolved)
%
% please see spm_MDP_VB
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% Philipp Schwartenbeck

% graphics
%==========================================================================
% col   = {'.b','.y','.g','.r','.c','.k'};
col   = {[0, 0.4470, 0.7410], ...       % blue
         [0.4940, 0.1840, 0.5560], ...  % purple
         [0.4660, 0.6740, 0.1880], ...  % green
         [0.9350, 0.1780, 0.2840], ...  % red
         [0.3010, 0.7450, 0.9330], ...  % cyan
         [0, 0, 0]};                    % black
cols  = [0:1/32:1; 0:1/32:1; 0:1/32:1]';

n_trials   = size(MDP,2);               % number of trials
n_timestep = size(MDP(1).V,1) + 1;      % number of time steps per trial

% MarkerSize = [24 24 24 24 24 24];
MarkerSize = 16;

for i = 1:n_trials
    
    % assemble performance
    %----------------------------------------------------------------------
    p(i)  = 0;
    
    for g = 1:numel(MDP(1).A)
        
        U = spm_softmax(MDP(i).C{g});
        
        for t = 1:n_timestep
            p(i) = p(i) + log(U(MDP(i).o(g,t),t))/n_timestep; % utility of outcomes over time steps
        end
        
    end
    
    o(:,i) = MDP(i).o(:,end); % observation
    u(:,i) = MDP(i).R(:,end); % chosen action
    
end

% Initial states and expected policies
%--------------------------------------------------------------------------
choice_prob = zeros(size(MDP(i).P,1),n_trials);
for i = 1:n_trials 
    choice_prob(:,i) = MDP(i).P;
end

t     = 1:n_trials;

% plot choices and beliefs about choices
subplot(4,1,1)

imagesc([1 - choice_prob]); colormap(cols) , hold on
chosen_action = [1 2 3]*u; plot([chosen_action],'.','MarkerSize',MarkerSize,'Color',col{1})
% don't plot choice conflict for now - it's already there in the shades of 
% grey of beliefs about choices
% choice_conflict = sum(choice_prob.*log(choice_prob));
% choice_conflict = -choice_conflict;
% choice_conflict = max(chosen_action) - choice_conflict;
% 
% plot(choice_conflict,'.c','MarkerSize',16), hold on
% plot(choice_conflict,':c')

title('Initial state and policy selection')
xlim([-1,n_trials+1])
set(gca, 'XTick', [0:n_trials]), 
set(gca, 'YTick', [1:3]), set(gca, 'YTickLabel', {'Stay','Safe','Risky'})
xlabel('Trial'),ylabel('Policy')


hold off

% Plot outcomes and their utilities
subplot(4,1,2)

bar(p,'k'), % utility of outcome
hold on
    
for i = 1:max(o(2,:)) % first row of o: risky (3) or safe (2) outcome, second row: reward (3=high r, 3=no r, 2=low r)
    j = find(o(2,:) == i);
    plot(t(j),ones(1,length(t(j)))*3,'.','MarkerSize',MarkerSize,'Color',col{rem(i - 1,6)+ 1})
end

title('Final outcome and performance')
xlim([-1,n_trials+1])
ylim([min(floor(p)),-min(floor(p))])
xlabel('Trial'),ylabel('Utility Outcomes'), set(gca, 'XTick', [0:n_trials])
hold off

