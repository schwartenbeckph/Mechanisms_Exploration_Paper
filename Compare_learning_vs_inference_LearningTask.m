% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing
%==========================================================================

% Compare performance of active learning, active inference and random
% behaviour

% Performance measure: how often reward obtained in high reward context at
% given time-step
%==========================================================================

function Compare_learning_vs_inference_LearningTask(simulate,saved_data)

N_sims = 100;

n_trials    = 32;

if simulate

    Rprobs      = [0.85 0.15];          % high or low reward
    
    alphas      = [8 8 1];              % random or precise agent
    ambiguities = [true false false];   % active inference agent on-off
    curiosities = [false true false];   % active learning agent on-off
      
    eta       = 0.5;       % Learning Rate
    beta      = 2^0;       % precision of policy selection
    
    for idx_Rprob = 1:length(Rprobs)
        
        Rprob     = Rprobs(idx_Rprob);      % probability for receiving reward in risky option
        
        for idx_agent = 1:length(alphas)

            alpha     = alphas(idx_agent);           % precision of action selection

            curiosity = curiosities(idx_agent);      % goal-directed exploration of parameters

            ambiguity = ambiguities(idx_agent);      % goal-directed exploration of states

            %% prelim - define generative model
            % outcome probabilities: A
            %--------------------------------------------------------------------------
            % We start by specifying the probabilistic mapping from hidden states
            % to outcomes; where outcome can be exteroceptive or interoceptive: The
            % exteroceptive outcomes A{1} provide cues about location,
            % while interoceptive outcome A{2) denotes different levels of reward
            %--------------------------------------------------------------------------
            % play around with different values and see what happens
            p          = Rprob;
            q          = 1 - Rprob;

            % Factor 1: Location, exteroceptive - no uncertainty about location
            %--------------------------------------------------------------------------
            % Not interesting for task
            A{1} = [1 0 0;         % starting position
                    0 1 0;         % safe option
                    0 0 1];        % risky option

            % Factor 2: Reward, interooceptive - uncertainty about reward prob
            %--------------------------------------------------------------------------
            % That's were true reward prob comes in
            A{2} = [1 0 0          % reward neutral  (starting position)
                    0 1 0          % low reward      (safe option)
                    0 0 p          % high reward     (risky option)
                    0 0 q];        % negative reward (risky option)

            % beliefs about outcome (likelihood) mapping
            %--------------------------------------------------------------------------
            a{1} = A{1}*128;
            
            % That's where learning comes in - start with uniform prior
            a{2} = [1 0 0          % reward neutral  (starting position)
                    0 1 0          % low reward      (safe option)
                    0 0 1/4       % high reward     (risky option)
                    0 0 1/4];     % negative reward (risky option)
                
            % controlled transitions: B{u}
            %--------------------------------------------------------------------------
            % Next, we have to specify the probabilistic transitions of hidden states
            % for each factor. Here, there are three actions taking the agent directly
            % to each of the three locations.
            %--------------------------------------------------------------------------
            B{1}(:,:,1) = [1 1 1; 0 0 0;0 0 0];     % move to the starting point
            B{1}(:,:,2) = [0 0 0; 1 1 1;0 0 0];     % move to safe  option (and check for reward)
            B{1}(:,:,3) = [0 0 0; 0 0 0;1 1 1];     % move to risky option (and check for reward)

            % priors: (utility) C
            %--------------------------------------------------------------------------
            % Finally, we have to specify the prior preferences in terms of log
            % probabilities over outcomes. Here, the agent prefers high rewards over
            % low rewards over no rewards
            %--------------------------------------------------------------------------
            C{1}  = [0 0 0]'; % preference for first factor - doesn't care

            cs = 2^1; % preference for safe option
            cr = 2^2; % preference for risky option win

            C{2}  = [0 cs cr -cs]'; % preference for: [staying at starting point | safe | risky + reward | risky + no reward]

            % now specify prior beliefs about initial states, in terms of counts. Here,
            % the hidden states are factorised into location and context:
            %--------------------------------------------------------------------------
            D{1}  = [1 0 0]'; % prior over starting point - rat 'starts' at starting point (not at safe or risky option

            % allowable policies (of depth T).  These are just sequences of actions
            % (with an action for each hidden factor)
            %--------------------------------------------------------------------------
            V     = [1 2 3]; % stay, go left, go right

            % MDP Structure
            % nothing to change here for now
            %==========================================================================
            mdp.V = V;                    % allowable policies
            mdp.A = A;                    % observation process
            mdp.a = a;                % observation model
            mdp.B = B;                    % transition probabilities
            mdp.C = C;                    % preferred states
            mdp.D = D;                    % prior over initial states
            mdp.s = 1;


            mdp.Aname = {'exteroceptive','interoceptive'};
            mdp.Bname = {'position'};

            mdp.alpha      = alpha;            % precision of action selection
            mdp.beta       = beta;             % precision of policy selection
            mdp.curiosity  = curiosity;        % Goal-directed exploration of parameters
            mdp.ambiguity  = ambiguity;        % Goal-directed exploration of states
            mdp.eta        = eta;              % Learning Rate

            Prob_risky = [];

            Reward = [];

            Trial_num = [];


            for trial = 1:N_sims

                elapsed_time = 0;
                tic

                % true parameters
                %--------------------------------------------------------------------------
                MDP        = mdp;
                [MDP(1:n_trials)] = deal(MDP);


                % solve and show behaviour over trials (and epistemic learning)
                %--------------------------------------------------------------------------
                MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

                choice_prob = extractfield(MDP, 'P'); choice_prob = reshape(choice_prob,3,length(choice_prob)/3);
                prob_risky  = choice_prob(3,:);

                reward            = extractfield(MDP, 'o'); reward = reshape(reward,4,length(reward)/4); 
                reward            = reward(4,:);
                reward(reward==2) = 1; % safe reward
                reward(reward==4) = 0;
                reward(reward==3) = 4; % risky reward

                Reward     = [Reward;reward'];
                Prob_risky = [Prob_risky;prob_risky'];
                Trial_num  = [Trial_num; [1:n_trials]'];

                elapsed_time = toc;

                fprintf('###trial %d of %d done, time needed: %d###\n',trial,N_sims,elapsed_time)

            end
            
            % creat 4-dim data array:
            % 1st: n_trials*N_sims
            % 2nds: three variables of interest
            % 3rd: agent (active inference, active learning, random)
            % 4th: reward (high or low)
            data_Fig12(:,1:3,idx_agent,idx_Rprob) = [Prob_risky,Trial_num,Reward];
            
            clear('MDP')
            clear('mdp')
            clear('A')
            clear('a')
            clear('B')
            clear('C')
            clear('D')
            clear('V')
        
        end
    
    end
    
else
    
    load(saved_data{1})

end

%% Plot probS to choose risky as a function of trials:

% Inference, high reward
data = data_Fig12(:,:,1,1);
Prob_risky = data(:,1);Trial_num = data(:,2);Reward = data(:,3);

prob_risky_trialNum_AI = splitapply(@mean,Prob_risky,Trial_num);
prob_risky_trialNum_AI_std = splitapply(@std,Prob_risky,Trial_num); 
prob_risky_trialNum_AI_se = prob_risky_trialNum_AI_std./sqrt(N_sims);

reward_trialNum_AI = splitapply(@mean,Reward,Trial_num);
reward_trialNum_AI_std = splitapply(@std,Reward,Trial_num); 
reward_trialNum_AI_se = reward_trialNum_AI_std./sqrt(N_sims);

% Learning, high reward
data = data_Fig12(:,:,2,1);
Prob_risky = data(:,1);Trial_num = data(:,2);Reward = data(:,3);

prob_risky_trialNum_AL = splitapply(@mean,Prob_risky,Trial_num);
prob_risky_trialNum_AL_std = splitapply(@std,Prob_risky,Trial_num); 
prob_risky_trialNum_AL_se = prob_risky_trialNum_AL_std./sqrt(N_sims);

reward_trialNum_AL = splitapply(@mean,Reward,Trial_num);
reward_trialNum_AL_std = splitapply(@std,Reward,Trial_num); 
reward_trialNum_AL_se = reward_trialNum_AL_std./sqrt(N_sims);

% Random, high reward
data = data_Fig12(:,:,3,1);
Prob_risky = data(:,1);Trial_num = data(:,2);Reward = data(:,3);

prob_risky_trialNum_R = splitapply(@mean,Prob_risky,Trial_num);
prob_risky_trialNum_R_std = splitapply(@std,Prob_risky,Trial_num); 
prob_risky_trialNum_R_se = prob_risky_trialNum_R_std./sqrt(N_sims);

reward_trialNum_R = splitapply(@mean,Reward,Trial_num);
reward_trialNum_R_std = splitapply(@std,Reward,Trial_num); 
reward_trialNum_R_se = reward_trialNum_R_std./sqrt(N_sims);

figure,hold on
plot(prob_risky_trialNum_AI,'-.','LineWidth',2)
% errorbar(prob_risky_trialNum_AI,prob_risky_trialNum_AI_se,'-.','LineWidth',2)
plot(prob_risky_trialNum_AL,'-.','LineWidth',2)
% errorbar(prob_risky_trialNum_AL,prob_risky_trialNum_AL_se,'-.','LineWidth',2)
plot(prob_risky_trialNum_R,'-.','LineWidth',2)
% errorbar(prob_risky_trialNum_R,prob_risky_trialNum_R_se,'-.','LineWidth',2)
title('Probability to choose risky option, high reward probability'), set(gcf,'color','white')
set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
ylim([-0.2,1])
set(gca, 'XTick', [0:2:length(prob_risky_trialNum_AI)]),xlabel('Trial Number')
legend({'State Exploration','Parameter Exploration','Random Exploration'})

figure,hold on
plot(cumsum(reward_trialNum_AI),'-.','LineWidth',2)
% errorbar(cumsum(reward_trialNum_AI),reward_trialNum_AI_se,'-.','LineWidth',2)
plot(cumsum(reward_trialNum_AL),'-.','LineWidth',2)
% errorbar(cumsum(reward_trialNum_AL),reward_trialNum_AL_se,'-.','LineWidth',2)
plot(cumsum(reward_trialNum_R),'-.','LineWidth',2)
% errorbar(cumsum(reward_trialNum_R),reward_trialNum_R_se,'-.','LineWidth',2)
title('Cumulative reward, high reward probability'), set(gcf,'color','white')
% set(gca, 'YTick', [0:0.25:1]),
ylabel('Reward (Pellets)')
ylim([0,128])
set(gca, 'XTick', [0:2:length(reward_trialNum_AI)]),xlabel('Trial Number')
legend({'State Exploration','Parameter Exploration','Random Exploration'})

% Inference, low reward
data = data_Fig12(:,:,1,2);
Prob_risky = data(:,1);Trial_num = data(:,2);Reward = data(:,3);

prob_risky_trialNum_AI = splitapply(@mean,Prob_risky,Trial_num);
prob_risky_trialNum_AI_std = splitapply(@std,Prob_risky,Trial_num); 
prob_risky_trialNum_AI_se = prob_risky_trialNum_AI_std./sqrt(N_sims);

reward_trialNum_AI = splitapply(@mean,Reward,Trial_num);
reward_trialNum_AI_std = splitapply(@std,Reward,Trial_num); 
reward_trialNum_AI_se = reward_trialNum_AI_std./sqrt(N_sims);

% Learning, low reward
data = data_Fig12(:,:,2,2);
Prob_risky = data(:,1);Trial_num = data(:,2);Reward = data(:,3);

prob_risky_trialNum_AL = splitapply(@mean,Prob_risky,Trial_num);
prob_risky_trialNum_AL_std = splitapply(@std,Prob_risky,Trial_num); 
prob_risky_trialNum_AL_se = prob_risky_trialNum_AL_std./sqrt(N_sims);

reward_trialNum_AL = splitapply(@mean,Reward,Trial_num);
reward_trialNum_AL_std = splitapply(@std,Reward,Trial_num); 
reward_trialNum_AL_se = reward_trialNum_AL_std./sqrt(N_sims);

% Random, low reward
data = data_Fig12(:,:,3,2);
Prob_risky = data(:,1);Trial_num = data(:,2);Reward = data(:,3);

prob_risky_trialNum_R = splitapply(@mean,Prob_risky,Trial_num);
prob_risky_trialNum_R_std = splitapply(@std,Prob_risky,Trial_num); 
prob_risky_trialNum_R_se = prob_risky_trialNum_R_std./sqrt(N_sims);

reward_trialNum_R = splitapply(@mean,Reward,Trial_num);
reward_trialNum_R_std = splitapply(@std,Reward,Trial_num); 
reward_trialNum_R_se = reward_trialNum_R_std./sqrt(N_sims);

figure,hold on
plot(prob_risky_trialNum_AI,'-.','LineWidth',2)
% errorbar(prob_risky_trialNum_AI,prob_risky_trialNum_AI_se,'-.','LineWidth',2)
plot(prob_risky_trialNum_AL,'-.','LineWidth',2)
% errorbar(prob_risky_trialNum_AL,prob_risky_trialNum_AL_se,'-.','LineWidth',2)
plot(prob_risky_trialNum_R,'-.','LineWidth',2)
% errorbar(prob_risky_trialNum_R,prob_risky_trialNum_R_se,'-.','LineWidth',2)
title('Probability to choose risky option, low reward probability'), set(gcf,'color','white')
set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
ylim([-0.2,1])
set(gca, 'XTick', [0:2:length(prob_risky_trialNum_AI)]),xlabel('Trial Number')
legend({'State Exploration','Parameter Exploration','Random Exploration'})

figure,hold on
plot(cumsum(reward_trialNum_AI),'-.','LineWidth',2)
% errorbar(cumsum(reward_trialNum_AI),reward_trialNum_AI_se,'-.','LineWidth',2)
plot(cumsum(reward_trialNum_AL),'-.','LineWidth',2)
% errorbar(cumsum(reward_trialNum_AL),reward_trialNum_AL_se,'-.','LineWidth',2)
plot(cumsum(reward_trialNum_R),'-.','LineWidth',2)
% errorbar(cumsum(reward_trialNum_R),reward_trialNum_R_se,'-.','LineWidth',2)
title('Cumulative reward, low reward probability'), set(gcf,'color','white')
% set(gca, 'YTick', [0:0.25:1]),
ylabel('Reward (Pellets)')
ylim([0,128])
set(gca, 'XTick', [0:2:length(reward_trialNum_AI)]),xlabel('Trial Number')
legend({'State Exploration','Parameter Exploration','Random Exploration'})

end

