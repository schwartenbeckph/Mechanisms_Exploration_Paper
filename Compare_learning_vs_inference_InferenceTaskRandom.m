% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing
%==========================================================================

% Compare performance of active learning, active inference and random
% behaviour

% Performance measure: how often reward obtained in high reward context at
% given time-step
%==========================================================================

function Compare_learning_vs_inference_InferenceTaskRandom(simulate,saved_data)

N_sims = 100;

n_trials    = 32;

Rprob = 0.85;

if simulate
    
    alphas      = [8 8 1];              % random or precise agent
    ambiguities = [true false false];   % active inference agent on-off
    curiosities = [false true false];   % active learning agent on-off
     
    eta       = 0.5;       % Learning Rate
    beta      = 2^0;       % precision of policy selection
    
        
    for idx_agent = 1:length(alphas)

        alpha     = alphas(idx_agent);           % precision of action selection

        curiosity = curiosities(idx_agent);      % goal-directed exploration of parameters

        ambiguity = ambiguities(idx_agent);      % goal-directed exploration of states

        %% prelim - define MDP
        % rng('default')
        rng('shuffle')

        % outcome probabilities: A
        %--------------------------------------------------------------------------
        % We start by specifying the probabilistic mapping from hidden states
        % to outcomes.
        %-------------------------------------------------------------------------
        a = Rprob;
        b = 1 - a;

        c = Rprob;
        d = 1 - c;

        if curiosity

            A{1}   = [1 1 0 0 0   0 0;    % ambiguous starting position (centre)
                      0 0 1 1 0   0 0;    % safe arm selected and rewarded
                      0 0 0 0 0.5 0 0;    % risky arm selected and rewarded
                      0 0 0 0 0.5 0 0;    % risky arm selected and not rewarded
                      0 0 0 0 0   1 0;    % informative cue - high reward prob
                      0 0 0 0 0   0 1];   % informative cue - low reward prob

        else

            A{1}   = [1 1 0 0 0 0 0 0;    % ambiguous starting position (centre)
                      0 0 1 1 0 0 0 0;    % safe arm selected and rewarded
                      0 0 0 0 a d 0 0;    % risky arm selected and rewarded
                      0 0 0 0 b c 0 0;    % risky arm selected and not rewarded
                      0 0 0 0 0 0 1 0;    % informative cue - high reward prob
                      0 0 0 0 0 0 0 1];   % informative cue - low reward prob

        end

        clear('a'), clear('b'),clear('c'),clear('d')  

        if curiosity
            a{1}   = [1 1 0 0 0   0 0;    % ambiguous starting position (centre)
                      0 0 1 1 0   0 0;    % safe arm selected and rewarded
                      0 0 0 0 1/4 0 0;    % risky arm selected and rewarded
                      0 0 0 0 1/4 0 0;    % risky arm selected and not rewarded
                      0 0 0 0 0   1 0;    % informative cue - high reward prob
                      0 0 0 0 0   0 1];   % informative cue - low reward prob
        end

        % controlled transitions: B{u}
        %--------------------------------------------------------------------------
        % Next, we have to specify the probabilistic transitions of hidden states
        % under each action or control state. Here, there are four actions taking the
        % agent directly to each of the four locations.
        %--------------------------------------------------------------------------
        if curiosity

            B{1}(:,:,1) = [1 0 0 0 0 1 0;  % go to middle    
                           0 1 0 0 0 0 1;    
                           0 0 1 0 0 0 0;    
                           0 0 0 1 0 0 0;    
                           0 0 0 0 1 0 0;    
                           0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0];

            B{1}(:,:,2) = [0 0 0 0 0 0 0;  % go safe    
                           0 0 0 0 0 0 0;    
                           1 0 1 0 0 1 0;    
                           0 1 0 1 0 0 1;    
                           0 0 0 0 1 0 0;    
                           0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0];

            B{1}(:,:,3) = [0 0 0 0 0 0 0;  % go risky    
                           0 0 0 0 0 0 0;    
                           0 0 1 0 0 0 0;    
                           0 0 0 1 0 0 0;    
                           1 0 0 0 1 1 1;    
                           0 0 0 0 0 0 0;
                           0 0 0 0 0 0 0];

            B{1}(:,:,4) = [0 0 0 0 0 0 0;  % go to cue    
                           0 0 0 0 0 0 0;    
                           0 0 1 0 0 0 0;    
                           0 0 0 1 0 0 0;    
                           0 0 0 0 1 0 0;    
                           1 0 0 0 0 1 0;
                           0 1 0 0 0 0 1];               
        else

            b{1}(:,:,1)  = [1 0 0 1; 0 1 0 0;0 0 1 0;0 0 0 0];     % move to/stay in the middle
            b{1}(:,:,2)  = [0 0 0 0; 1 1 0 1;0 0 1 0;0 0 0 0];     % move up left to safe  (and check for reward)
            b{1}(:,:,3)  = [0 0 0 0; 0 1 0 0;1 0 1 1;0 0 0 0];     % move up right to risky (and check for reward)
            b{1}(:,:,4)  = [0 0 0 0; 0 1 0 0;0 0 1 0;1 0 0 1];     % move down (check cue)

            for i = 1:4
                B{1}(:,:,i) = kron(b{1}(:,:,i),eye(2));
            end
            clear('b')

        end

        % priors: (utility) C
        %--------------------------------------------------------------------------
        % Finally, we have to specify the prior preferences in terms of log
        % probabilities. Here, the agent prefers rewarding outcomes
        %--------------------------------------------------------------------------
        cs = 2^1; % preference for safe option
        cr = 2^2; % preference for risky option win

        C{1}  = [0 cs cr -cs 0 0]'; % preference for: [staying at starting point | safe | risky + reward | risky + no reward | cue context 1 | cue context 2]

        % now specify prior beliefs about initial state, in terms of counts
        %--------------------------------------------------------------------------
        if curiosity
            D{1}  = [1/4 1/4 0 0 0 0 0]';
        else
            D{1}  = kron([1/4 0 0 0],[1 1])';
        end
        % allowable policies (of depth T).  These are just sequences of actions
        %--------------------------------------------------------------------------
        V  = [1  1  1  1  2  3  4  4  4  4
              1  2  3  4  2  3  1  2  3  4];
          
        %% MDP Structure - this will be used to generate arrays for multiple trials
        %==========================================================================
        mdp.V = V;                    % allowable policies
        mdp.A = A;                    % observation model

        if curiosity
            mdp.a = a;
        end

        mdp.B = B;                    % transition probabilities
        mdp.C = C;                    % preferred states
        mdp.D = D;                    % prior over initial states
        mdp.s = 1;                    % initial state - high reward context

        mdp.alpha      = alpha;            % precision of action selection
        mdp.beta       = beta;             % precision of policy selection
        mdp.curiosity  = curiosity;        % Goal-directed exploration of parameters
        mdp.ambiguity  = ambiguity;        % Goal-directed exploration of states
        mdp.eta        = eta;              % Learning Rate

        Prob_cue = [];

        Reward = [];

        Trial_num = [];

        for trial = 1:N_sims

            elapsed_time = 0;
            tic

            % true parameters
            %--------------------------------------------------------------------------
            i = rand(1,n_trials) > 1/2;

            MDP = mdp;

            [MDP(1:n_trials)] = deal(MDP);
            [MDP(i).s]        = deal(2);  % randomise context

            % solve and show behaviour over trials (and epistemic learning)
            %--------------------------------------------------------------------------
            MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

            choice_prob                   = extractfield(MDP, 'P'); 
            choice_prob                   = reshape(choice_prob,4,length(choice_prob)/4); 
            choice_prob(:,2:2:n_trials*2) = [];
            
            prob_cue = choice_prob(4,:);

            Prob_cue = [Prob_cue;prob_cue'];

            reward            = extractfield(MDP, 'o'); 
            reward            = reshape(reward,3,length(reward)/3); 
            reward            = reward(3,:);
            reward(reward==1) = 0;
            reward(reward==5) = 0;
            reward(reward==6) = 0;
            reward(reward==2) = 1; % safe reward
            reward(reward==4) = 0;
            reward(reward==3) = 4; % risky reward

            Reward = [Reward;reward'];

            Trial_num = [Trial_num; [1:n_trials]'];

            elapsed_time = toc;

            fprintf('###trial %d of %d done, time needed: %d###\n',trial,N_sims,elapsed_time)

        end
        
        % creat 3-dim data array:
        % 1st: n_trials*N_sims
        % 2nds: three variables of interest
        % 3rd: agent (active inference, active learning, random)
        data_Fig13(:,1:3,idx_agent) = [Prob_cue,Trial_num,Reward];
        
        clear('MDP')
        clear('mdp')
        clear('A')
        clear('a')
        clear('B')
        clear('C')
        clear('D')
        clear('V')
    
    end
    
else
    
    load(saved_data{1})
    
end

%% Plot probS to choose cue and reward as a function of trials:

% active inference
data = data_Fig13(:,:,1);
Prob_cue = data(:,1);Trial_num = data(:,2);Reward = data(:,3);
prob_cue_trialNum_AI = splitapply(@mean,Prob_cue,Trial_num);
reward_trialNum_AI = splitapply(@mean,Reward,Trial_num);

% active learning
data = data_Fig13(:,:,2);
Prob_cue = data(:,1);Trial_num = data(:,2);Reward = data(:,3);
prob_cue_trialNum_AL = splitapply(@mean,Prob_cue,Trial_num);
reward_trialNum_AL = splitapply(@mean,Reward,Trial_num);

% random agent
data = data_Fig13(:,:,3);
Prob_cue = data(:,1);Trial_num = data(:,2);Reward = data(:,3);
prob_cue_trialNum_R = splitapply(@mean,Prob_cue,Trial_num);
reward_trialNum_R = splitapply(@mean,Reward,Trial_num);

figure,hold on
plot(prob_cue_trialNum_AI,'-.','LineWidth',2)
plot(prob_cue_trialNum_AL,'-.','LineWidth',2)
plot(prob_cue_trialNum_R,'-.','LineWidth',2)
title('Probability to sample the cue first as a function of time'), set(gcf,'color','white')
set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
ylim([-0.2,1])
set(gca, 'XTick', [0:2:length(prob_cue_trialNum_AI)]),xlabel('Trial Number')
legend({'State Exploration','Parameter Exploration','Random Exploration'})

figure,hold on
plot(cumsum(reward_trialNum_AI),'-.','LineWidth',2)
plot(cumsum(reward_trialNum_AL),'-.','LineWidth',2)
plot(cumsum(reward_trialNum_R),'-.','LineWidth',2)
title('Cumulative reward'), set(gcf,'color','white')
ylabel('Reward (Pellets)')
ylim([0,128])
set(gca, 'XTick', [0:2:length(reward_trialNum_R)]),xlabel('Trial Number')
legend({'State Exploration','Parameter Exploration','Random Exploration'})


end


