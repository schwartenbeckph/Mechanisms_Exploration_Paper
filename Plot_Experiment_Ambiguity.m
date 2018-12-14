function [] = Plot_Experiment_Ambiguity(MDP,mdp)

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

% col         = {'b.','m.','g.','r.','c.','k.'};
col_context = {[0.1, 0.5, 0],[0.5, 0.1, 0]};


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
    s(:,i) = MDP(i).s(:,1);
    o(:,i) = MDP(i).o(:,end); % observation
    u(:,i) = MDP(i).R(:,end); % chosen action
    
end

% Initial states and expected policies
%--------------------------------------------------------------------------
choice_prob = zeros(size(MDP(i).P,1),n_trials);
for i = 1:n_trials 
    choice_prob(:,i) = MDP(i).P(:,1);
end

t     = 1:n_trials;

% plot choices and beliefs about choices
subplot(4,1,1)

imagesc((1 - choice_prob))
colormap(cols)
hold on

% choice_conflict = sum(choice_prob.*log(choice_prob+eps));

chosen_action = [1  1  1  1  2  3  4  4  4  4]*u; 
plot(chosen_action,'.','MarkerSize',MarkerSize,'Color',col{1})

title('Policy selection')
xlim([-1,n_trials+1])
if n_trials<33
    set(gca, 'XTick', [0:n_trials])
else
    set(gca, 'XTick', [0:2:n_trials])
end
set(gca, 'YTick', 1:4), set(gca, 'YTickLabel', {'Stay','Safe','Risky' 'Cue'})
xlabel('Trial'),ylabel('Policy')
hold off

% Plot outcomes and their utilities
subplot(4,1,2)

bar(p,'k'), % utility of outcome
hold on
    
for g = 1%:Ng
    for i = 1:max(o(g,:))             % first row of o: risky (3) or safe (2) outcome, second row: reward (3=high r, 3=no r, 2=low r)
        j = find(o(g,:) == i);
        plot(t(j),j - j + 1 + g,'.','MarkerSize',MarkerSize,'Color',col{rem(i - 1,6)+ 1})
    end
    for i = 1:max(s(g,:))             % first row of o: risky (3) or safe (2) outcome, second row: reward (3=high r, 3=no r, 2=low r)
        j = find(s(g,:) == i);
        plot(t(j),j - j,'.','Color',col_context{i},'MarkerSize',MarkerSize)
    end
end

title('Current Context and final outcome')
xlim([-1,n_trials+1])
ylim([min(floor(p)),-min(floor(p))])
xlabel('Trial'),ylabel('Utility Outcomes')
if n_trials<33
    set(gca, 'XTick', [0:n_trials])
else
    set(gca, 'XTick', [0:2:n_trials])
end
hold off

subplot(4,1,3)

d_con = zeros(2,n_trials+1);
d_fig = zeros(2,n_trials+1);

d_con(:,1) = mdp.d{1}(1:2);
d_fig(:,1) = mdp.d{1}(1:2);
d_fig(:,1) = d_fig(:,1)/sum(d_fig(:,1));
for i = 1:n_trials
    d_con(:,i+1) = MDP(i).d{1}(1:2);
    d_fig(:,i+1) = MDP(i).d{1}(1:2);
    d_fig(:,i+1) = d_fig(:,i+1)/sum(d_fig(:,i+1));
end

% a_con = sum(a_con)/max(sum(a_con));

plot(d_fig(1,:),'.','MarkerSize',MarkerSize,'Color',col{3}), hold on
plot(d_fig(1,:),':','Color',col{3})

plot(d_fig(2,:),'.','MarkerSize',MarkerSize,'Color',col{4})
plot(d_fig(2,:),':','Color',col{4})

% uncertainty_risky = -sum(a_fig.*log(a_fig));

% plot(uncertainty_risky,'.c','MarkerSize',12)
% plot(uncertainty_risky,':c')

title('Beliefs and certainty about context (high/low reward)')
xlim([0,n_trials+2])
ylim([0,1.2])
xlabel('Trial'),ylabel('Beliefs')
if n_trials<33
    set(gca, 'XTick', [1:n_trials+1])
else
    set(gca, 'XTick', [1:2:n_trials+1])
end
set(gca, 'XTickLabel', [0:n_trials])
hold off

subplot(4,1,4)

plot(d_con(1,:),'.','MarkerSize',12,'Color',col{3})
hold on
plot(d_con(1,:),':','Color',col{3})

plot(d_con(2,:),'.','MarkerSize',12,'Color',col{4})
plot(d_con(2,:),':','Color',col{4})

title('Concentration parameters context (high/low reward)')
xlim([0,n_trials+2])
ylim([0,max(max(d_con))+1])
xlabel('Trial'),ylabel('Value')
if n_trials<33
    set(gca, 'XTick', [1:n_trials+1])
else
    set(gca, 'XTick', [1:2:n_trials+1])
end
set(gca, 'XTickLabel', [0:n_trials])
hold off

