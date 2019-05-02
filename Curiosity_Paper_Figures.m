% Simulations of active inference and active learning

clear 
close all

% rng('default')
rng('shuffle')

based = pwd;
data_dir = fullfile(based,'SomeData');

cd(based)
addpath(data_dir)

% you also need to add spm12 and spm12\toolbox\DEM to your path

active_learning         = false;
active_learning_lots    = false;
active_inference        = false;
active_inference_lots   = false;
learning_vs_inference   = false;
effects_modelparams_a0  = false;
effects_modelparams_cL  = false;
effects_modelparams_cI  = false;
effects_modelparams_d   = false;
active_learning_sigmoid = false;
active_learning_value   = false;

%% ========================================================================
% 1.) Active Learning
% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing
%==========================================================================

if active_learning

    %%%% USED IN PAPER %%%%
    saved_data = {'MDP_learning_examplefull' ...                   % full examples, standard precision
                  'MDP_learning_examplefull_highR' ...             % full examples, standard precision, high reward prob
                  'MDP_learning_examplefull_imprecise' ...         % full examples, low precision
                  'MDP_learning_examplefull_precise' ...           % full examples, high precision
                  'MDP_learning_noCuriosity_highR' ...             % full examples, no curiosity, standard precision, high reward prob
                  'MDP_learning_noCuriosity_imprecise_highR'};     % full examples, no curiosity, low precision, high reward prob

    simulate = false;

    N_Figures   = 1:6;
    Rprobs      = [0.5 0.75 0.5 0.5 0.75 0.75];
    betas       = [2^0 2^0 2^-3 2^3 2^0 2^3];
    alphas      = [4 4 4 4 4 4];
    etas        = [0.5 0.5 0.5 0.5 0.5 0.5];
    curiosities = [true true true true false false];

    n_trials    = 32;

    for idx_figure = 1:length(N_Figures)

        Rprob     = Rprobs(idx_figure);       % probability for receiving reward in risky option
        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        curiosity = curiosities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_learning(Rprob,beta,alpha,eta,curiosity);
        
        if simulate
            
            MDP = mdp;

            [MDP(1:n_trials)] = deal(mdp);

            % solve and show behaviour over trials (and epistemic learning
            MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

        else

            load(saved_data{idx_figure})

        end

        % plot
        figure(N_Figures(idx_figure)), set(gcf,'color','white')
        if idx_figure<3
            Plot_Experiment_Curiosity(MDP,mdp);
        else
            Plot_Experiment_Curiosity_noCon(MDP,mdp);
        end

    end

end
%% ========================================================================
% 2.) Active Learning - simulate lots of trials
% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing
%==========================================================================

if active_learning_lots

    %%%% USED IN PAPER %%%%
    saved_data = {'data_sim_curiosity' ...            % simulations with curiosity
                  'data_sim_nocuriosity_random' ...   % simulations without curiosity
                  'data_sim_noLearning'};             % simulations without curiosity and learning

    simulate = false;

    N_Figures   = 1:3;
    Rprobs      = [0.5 0.5 0.5];
    betas       = [2^0 2^3 2^1];
    alphas      = [4 4 4];
    etas        = [0.5 0.5 0.5];
    curiosities = [true false false];
    learning    = [true true false];

    n_trials    = 32;

    N_sims = 1000;

    for idx_figure = 1:length(N_Figures)

        Rprob     = Rprobs(idx_figure);       % probability for receiving reward in risky option
        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        curiosity = curiosities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_learning(Rprob,beta,alpha,eta,curiosity);

        if ~learning(idx_figure)
           mdp = rmfield(mdp,'a'); 
        end

        if simulate

            Prob_risky = [];
            Trial_num = [];
            if learning(idx_figure)
                A_HR = [];
                A_NR = [];
            end

            for trial = 1:N_sims

                elapsed_time = 0;
                tic

                MDP = mdp;

                [MDP(1:n_trials)] = deal(mdp);

                % solve and show behaviour over trials (and epistemic learning
                MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

                choice_prob = extractfield(MDP, 'P'); choice_prob = reshape(choice_prob,3,length(choice_prob)/3);
                prob_risky = choice_prob(3,:);

                Prob_risky = [Prob_risky;prob_risky'];

                if learning(idx_figure)
                    a = extractfield(MDP, 'a');
                    a_HR = zeros(1,n_trials);
                    a_NR = zeros(1,n_trials);

                    for i = 1:n_trials
                        a_HR(i) = a{i}{2}(3,3);
                        a_NR(i) = a{i}{2}(4,3);
                    end

                    A_HR = [A_HR; a_HR'];
                    A_NR = [A_NR; a_NR'];
                end

                Trial_num = [Trial_num; [1:n_trials]'];

                elapsed_time = toc;

                fprintf('###trial %d of %d done, time needed: %d###\n',trial,N_sims,elapsed_time)

            end

            if learning(idx_figure)
                data = [A_HR,A_NR,Prob_risky,Trial_num];
            else
                data = [Prob_risky,Trial_num];
            end

        else

            load(saved_data{idx_figure})

        end

        if idx_figure < length(N_Figures)
        A_HR       = data(:,1); 
        A_NR       = data(:,2); 
        Prob_risky = data(:,3); 
        Trial_num  = data(:,4);
        else
            Prob_risky = data(:,1); 
            Trial_num  = data(:,2);
        end

        A_risky    = [A_HR A_NR];

        if idx_figure < length(N_Figures)
            % Plot prob to choose risky as a function of concentration params in A:
            [unique_A, ~, ib]   = unique(A_risky, 'rows');

            prob_risky_unique_A = splitapply(@mean,Prob_risky,ib);

            y_axis      = (unique_A+.25)/.5; 
            y_axis(:,2) = max(y_axis(:,2))-(y_axis(:,2)-1); % get y-axis right

            A_plot      = zeros(max(y_axis(:,2)),max(y_axis(:,1)));  % rows: y = concentration param no reward; colums: x = concentration parameter high reward

            ind         = sub2ind(size(A_plot), y_axis(:,2), y_axis(:,1));
            A_plot(ind) = prob_risky_unique_A;

            plot_size   = min(size(A_plot)); A_plot = A_plot(1:plot_size,1:plot_size);

            figure(N_Figures(idx_figure)), set(gcf,'color','white')
            imagesc(A_plot)
            title('Probability to choose risky option as a function of observation model','fontsize',13)
            set(gca, 'YTick', [1:2:plot_size]),set(gca, 'YTickLabel', {[(plot_size/2-1)*0.5+0.25:-0.5:0.25]}),ylabel('Concentration Parameter no reward','fontsize',14)
            set(gca, 'XTick', [1:2:plot_size]),set(gca, 'XTickLabel', {[0.25:0.5:(plot_size/2-1)*0.5+0.25]}),xlabel('Concentration Parameter high reward','fontsize',14)
            colorbar,caxis([0 1])

        end

        % Plot prob to choose risky as a function of trials:
        prob_risky_trialNum(idx_figure,:) = splitapply(@mean,Prob_risky,Trial_num)';
        
%         save(saved_data{idx_figure},'data')

    end
    
    figure(N_Figures(end)), hold on
    set(gcf,'color','white')
    plot(prob_risky_trialNum(1,:),'-.','LineWidth',2)
    plot(prob_risky_trialNum(2,:),'-.','LineWidth',2)
    plot(prob_risky_trialNum(3,:),'-.','LineWidth',2)
    title('Probability to choose risky option as a function of time','fontsize',15)
    set(gca, 'YTick', [0:0.25:1]), ylabel('Probability','fontsize',14),ylim([0,1])
    set(gca, 'XTick', [0:2:length(prob_risky_trialNum)]),xlabel('Trial Number','fontsize',14)
    legend('Active Learning','Random Exploration','No Learnnig')

end

%% ========================================================================
% 3.) Active Inference
% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing - or whether to sample a cue that is indicative of the reward
% probability of the risky option (which can be high or low).
%==========================================================================

if active_inference

    %%%% USED IN PAPER %%%%
    saved_data = {'MDP_inference_randomcontext' ...         % example of inference, random context
                  'MDP_inference_stablecontext' ...         % example of inference, stable context
                  'MDP_inferenceNo_stablecontext'};         % example of no inference, stable context

    simulate = false;

    N_Figures   = 1:3;
    betas       = [2^0 2^0 2^0];
    alphas      = [16 16 16];
    etas        = [0.5 0.5 0.5];
    ambiguities = [true true false];

    n_trials    = 32;

    for idx_figure = 1:length(N_Figures)

        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        ambiguity = ambiguities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_inference(beta,alpha,eta,ambiguity);
        
        if simulate
            
            MDP = mdp;

            [MDP(1:n_trials)] = deal(mdp);

            if idx_figure==1 % create random initial state
                i = rand(1,n_trials) > 1/2;
                [MDP(i).s] = deal(2);
            end

            % solve and show behaviour over trials (and epistemic learning
            MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

        else

            load(saved_data{idx_figure})

        end

        % plot
        figure(N_Figures(idx_figure)), set(gcf,'color','white')
        Plot_Experiment_Ambiguity_noCon(MDP,mdp);


    end

end

%% ========================================================================
% 4.) Active Inference - simulate lots of trials
% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing - or whether to sample a cue that is indicative of the reward
% probability of the risky option (which can be high or low).
%==========================================================================

if active_inference_lots

    %%%% USED IN PAPER %%%%
    saved_data = {'data_sim_state_ambiguity_random' ...            % example of inference, random context
                  'data_sim_state_ambiguity'};                     % example of inference, stable context

    simulate = false;

    N_Figures   = 1:2;
    betas       = [2^0 2^0 2^0];
    alphas      = [16 16 16];
    etas        = [0.5 0.5 0.5];
    ambiguities = [true true false];

    n_trials    = 32;

    N_sims = 1000;

    for idx_figure = 1:length(N_Figures)       

        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        ambiguity = ambiguities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_inference(beta,alpha,eta,ambiguity);

        if simulate

            Prob_cue = [];
            Trial_num = [];
            D_HR = [];
            D_LR = [];

            for trial = 1:N_sims

            elapsed_time = 0;
            tic
            
            MDP = mdp;

            [MDP(1:n_trials)] = deal(mdp);

            if idx_figure==1 % create random initial state
                i = rand(1,n_trials) > 1/2;
                [MDP(i).s] = deal(2);
            end

            % solve and show behaviour over trials (and epistemic learning
            MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

            choice_prob = extractfield(MDP, 'P'); 
            choice_prob = reshape(choice_prob,4,length(choice_prob)/4); 
            choice_prob(:,2:2:n_trials*2) = [];
            
            prob_cue = choice_prob(4,:);

            Prob_cue = [Prob_cue;prob_cue'];

            d = extractfield(MDP, 'd');
            d_HR = zeros(1,n_trials);
            d_LR = zeros(1,n_trials);

            for i = 1:n_trials
                d_HR(i) = d{i}{1}(1);
                d_LR(i) = d{i}{1}(2);
            end

            D_HR = [D_HR; d_HR'];
            D_LR = [D_LR; d_LR'];

            Trial_num = [Trial_num; [1:n_trials]'];

            elapsed_time = toc;

            fprintf('###trial %d of %d done, time needed: %d###\n',trial,N_sims,elapsed_time)

            end

            data = [D_HR,D_LR,Prob_cue,Trial_num];

        else

            load(saved_data{idx_figure})

        end

        D_HR = data(:,1); 
        D_LR = data(:,2); 
        Prob_cue = data(:,3); 
        Trial_num = data(:,4);

        % Plot prob to choose risky as a function of trials:
        prob_cue_trialNum(idx_figure,:) = splitapply(@mean,Prob_cue,Trial_num)';

    end
    
        figure, hold on 
        set(gcf,'color','white'), title('Probability to sample the cue first as a function of time')
        plot(prob_cue_trialNum(1,:),'-.','LineWidth',2)
        plot(prob_cue_trialNum(2,:),'-.','LineWidth',2)
        set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
        ylim([0,1.2])
        set(gca, 'XTick', [0:2:length(prob_cue_trialNum)]),xlabel('Trial Number') 
        legend('Random context','Stable context')
        hold off

end

%% ========================================================================
% 5.) Active inference vs. active learning
% Compare active learning (curiosity) and active inference (ambiguity) in
% the two tasks above, i.e. 
% i) unkown risky option (Figure 12)
% ii) cue random context (Figure 13)
% iii) cue stable context (Figure 14)
%==========================================================================


if learning_vs_inference

    %%%% USED IN PAPER %%%%           
    saved_data_I   = {'data_Fig12'};                   
                
    saved_data_II  = {'data_Fig13'};   
                
    saved_data_III = {'data_Fig14'};                 

    simulate = false;
   
    % plots are outsourced to keep it more readable:
    Compare_learning_vs_inference_LearningTask(simulate,saved_data_I)
    
    Compare_learning_vs_inference_InferenceTaskRandom(simulate,saved_data_II)
    
    Compare_learning_vs_inference_InferenceTaskStable(simulate,saved_data_III)
        
    
end

%% ========================================================================
% 6.) Illustrate effects of other parameters - effects of a0
% Can be illustrated using individual experiments
%==========================================================================

if effects_modelparams_a0

    % i) effects of a0 (in learning):

    %%%% USED IN PAPER %%%%
    saved_data = {'MDP_learning_a0_fulluncertainty' ...     % uncertainty about everything
                  'MDP_Learning_a0_startuncertainty' ...    % uncertainty about start position
                  'MDP_Learning_a0_safeuncertainty' ...     % uncertainty about safe
                  'MDP_Learning_a0_optimism' ...            % optimism about high reward in low reward world
                  'MDP_Learning_a0_pessimism' ...           % pessimism about high reward in high reward world
                  'MDP_Learning_a0_LowEta'};                % slower learning - longer information gain

    simulate = false;

    N_Figures   = 1:6;
    Rprobs      = [0.75 0.75 0.75 0.25 0.75 0.25];
    betas       = [2^0 2^0 2^0 2^0 2^0 2^0];
    alphas      = [4 4 4 4 4 4];
    etas        = [0.5 0.5 0.5 0.5 0.5 0.05];
    curiosities = [true true true true true true];

    n_trials    = 32;

    for idx_figure = 1:length(N_Figures)

        Rprob     = Rprobs(idx_figure);       % probability for receiving reward in risky option
        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        curiosity = curiosities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_learning(Rprob,beta,alpha,eta,curiosity);

        if idx_figure==1
            mdp.a{2}(1:4,1:3) = 1/4;
        elseif idx_figure==2
            mdp.a{2}(1:4,1) = 1/4;
        elseif idx_figure==3
            mdp.a{2}(1:4,2) = 1/4; 
        elseif idx_figure==4
            mdp.a{2}(3,3) = 8; 
        elseif idx_figure==5
            mdp.a{2}(4,3) = 2; 
        end
        
        if simulate
            
            MDP = mdp;

            [MDP(1:n_trials)] = deal(mdp);

            % solve and show behaviour over trials (and epistemic learning
            MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

        else

            load(saved_data{idx_figure})

        end

        % plot
        figure(N_Figures(idx_figure)), set(gcf,'color','white')
        Plot_Experiment_Curiosity_ParamsEffects(MDP,mdp);
    end
    
end
    
%% ========================================================================
% 7.) Illustrate effects of other parameters - effects of c in Learning
% Illustrates risk etc
% Best illustrated using lots of trials
%==========================================================================    

if effects_modelparams_cL
    

    %%%% USED IN PAPER %%%%
    saved_data = {'data_sim_learrning_NoRisky0' ...       % No risky reward = starting point (i.e. less bad)
                  'data_sim_learrning_NoRiskyBad' ...     % No risky reward is really bad
                  'data_sim_learrning_SafeEqualRisky' ... % Safe option is as good as risky option (i.e., risk averse agent)
                  'data_sim_learrning_RiskSeeking' ...    % Risky option really is much much better (i.e., risk seeking agent)
                  'data_sim_learrning_Uniform' ...        % Uniform preferences - should lead to avoidance of risky option because of ambiguity
                  'data_sim_learrning_Reference'};        % Reference as in Figure 7     

    simulate = false;

    N_Figures   = 1:6;
    Rprobs      = [0.5 0.5 0.5 0.5 0.5 0.5];
    betas       = [2^0 2^0 2^0 2^0 2^0 2^0];
    alphas      = [4 4 4 4 4 4];
    etas        = [0.5 0.5 0.5 0.5 0.5 0.5];
    curiosities = [true true true true true true];

    n_trials    = 32;

    N_sims = 1000;
    
    figure, set(gcf,'color','white'), hold on

    for idx_figure = 1:length(N_Figures)

        Rprob     = Rprobs(idx_figure);       % probability for receiving reward in risky option
        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        curiosity = curiosities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_learning(Rprob,beta,alpha,eta,curiosity);
        
        if idx_figure==1
            mdp.C{2}(4) = 0; 
        elseif idx_figure==2
            mdp.C{2}(4) = -4; 
        elseif idx_figure==3
            mdp.C{2}(2) = 4; 
        elseif idx_figure==4
            mdp.C{2}(3) = 8; 
        elseif idx_figure==5
            mdp.C{2}(1:4) = 0; 
        end

        if simulate

            Prob_risky = [];
            Trial_num = [];
            A_HR = [];
            A_NR = [];

            for trial = 1:N_sims

                elapsed_time = 0;
                tic

                MDP = mdp;

                [MDP(1:n_trials)] = deal(mdp);

                % solve and show behaviour over trials (and epistemic learning
                MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

                choice_prob = extractfield(MDP, 'P'); 
                choice_prob = reshape(choice_prob,3,length(choice_prob)/3);
                
                prob_risky = choice_prob(3,:);

                Prob_risky = [Prob_risky;prob_risky'];

                a = extractfield(MDP, 'a');
                a_HR = zeros(1,n_trials);
                a_NR = zeros(1,n_trials);

                for i = 1:n_trials
                    a_HR(i) = a{i}{2}(3,3);
                    a_NR(i) = a{i}{2}(4,3);
                end

                A_HR = [A_HR; a_HR'];
                A_NR = [A_NR; a_NR'];

                Trial_num = [Trial_num; [1:n_trials]'];

                elapsed_time = toc;

                fprintf('###trial %d of %d done, time needed: %d###\n',trial,N_sims,elapsed_time)

            end

            data = [A_HR,A_NR,Prob_risky,Trial_num];

        else

            load(saved_data{idx_figure})

        end

        A_HR       = data(:,1); 
        A_NR       = data(:,2); 
        Prob_risky = data(:,3); 
        Trial_num  = data(:,4);

        A_risky    = [A_HR A_NR];

        % Plot prob to choose risky as a function of trials:
        prob_risky_trialNum = splitapply(@mean,Prob_risky,Trial_num);

%         figure(N_Figures(idx_figure)), set(gcf,'color','white')
        plot(prob_risky_trialNum,'-.','LineWidth',2)
        title('Probability to choose risky option as a function of time')
        set(gca, 'YTick', [0:0.25:1]), ylabel('Probability'),ylim([0,1])
        set(gca, 'XTick', [0:2:length(prob_risky_trialNum)]),xlabel('Trial Number')
    
    end
    
    legend({'Indifferent no reward','Risk averse','Safe preferred','Risk seeking','Uniform preferences','Reference'})
    hold off

end

%% ========================================================================
% 8.) Illustrate effects of other parameters - effects of c in Inference
% Illustrates cost of information
% Can be illustrated using individual experiments
%========================================================================== 

if effects_modelparams_cI
    
    % i) Simulate single experiment with 64 trials
    
    %%%% USED IN PAPER %%%%
    saved_data = {'MDP_inference_C_InfoReward' ...     % stable context, cue is also (extrinsically) rewarding
                  'MDP_inference_C_InfoNeutral' ...    % stable context, cue is neutral
                  'MDP_inference_C_InfoCost'};         % stable context, cue has negative reward (higher cost)

    simulate = false;

    N_Figures   = 1:3;
    betas       = [2^0 2^0 2^0];
    alphas      = [16 16 16];
    etas        = [0.5 0.5 0.5];
    ambiguities = [true true true];

    n_trials    = 32;

    for idx_figure = 1:length(N_Figures)

        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        ambiguity = ambiguities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_inference(beta,alpha,eta,ambiguity);

        if idx_figure==1
            mdp.C{1}(5:6) = 0.5;
        elseif idx_figure==1
            mdp.C{1}(5:6) = 0;
        elseif idx_figure==1
            mdp.C{1}(5:6) = -0.5;
        end
        
        if simulate
            
            MDP = mdp;

            [MDP(1:n_trials)] = deal(mdp);

            % solve and show behaviour over trials (and epistemic learning
            MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

        else

            load(saved_data{idx_figure})

        end

        % plot
        figure(N_Figures(idx_figure)), set(gcf,'color','white')
        Plot_Experiment_Ambiguity_ParamsEffects(MDP,mdp);

    end
    
    
    % ii) Simulate lots of experiments with 32 trials
    
    %%%% USED IN PAPER %%%%
    saved_data = {'data_sim_state_ambiguity_neg05C'  ...     % lots of trials, cue quite costly
                  'data_sim_state_ambiguity_neg025C' ...     % lots of trials, cue a bit costly
                  'data_sim_state_ambiguity_0C'      ...     % lots of trials, cue 'neutral'
                  'data_sim_state_ambiguity_025C'    ...     % lots of trials, cue a bit rewarding
                  'data_sim_state_ambiguity_05C'};           % lots of trials, cue quite rewarding

    simulate = false;

    N_Figures   = 1:5;
    betas       = [2^0 2^0 2^0 2^0 2^0];
    alphas      = [16 16 16 16 16];
    etas        = [0.5 0.5 0.5 0.5 0.5];
    ambiguities = [true true true true true];

    n_trials    = 32;

    N_sims = 1000;
    
    figure, set(gcf,'color','white'), title('Probability to sample the cue first as a function of time')
    hold on

    for idx_figure = 1:length(N_Figures)

        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        ambiguity = ambiguities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_inference(beta,alpha,eta,ambiguity);
        
        if idx_figure==1
            mdp.C{1}(5:6) = -0.5;
        elseif idx_figure==2
            mdp.C{1}(5:6) = -0.25;
        elseif idx_figure==3
            mdp.C{1}(5:6) = 0;
        elseif idx_figure==4
            mdp.C{1}(5:6) = 0.25;
        elseif idx_figure==5
            mdp.C{1}(5:6) = 0.5;            
        end        

        if simulate

            Prob_cue = [];
            Trial_num = [];
            D_HR = [];
            D_LR = [];

            for trial = 1:N_sims

                elapsed_time = 0;
                tic

                MDP = mdp;

                [MDP(1:n_trials)] = deal(mdp);

                % solve and show behaviour over trials (and epistemic learning
                MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

                choice_prob = extractfield(MDP, 'P'); 
                choice_prob = reshape(choice_prob,4,length(choice_prob)/4); 
                choice_prob(:,2:2:n_trials*2) = [];

                prob_cue = choice_prob(4,:);

                Prob_cue = [Prob_cue;prob_cue'];

                d = extractfield(MDP, 'd');
                d_HR = zeros(1,n_trials);
                d_LR = zeros(1,n_trials);

                for i = 1:n_trials
                    d_HR(i) = d{i}{1}(1);
                    d_LR(i) = d{i}{1}(2);
                end

                D_HR = [D_HR; d_HR'];
                D_LR = [D_LR; d_LR'];

                Trial_num = [Trial_num; [1:n_trials]'];

                elapsed_time = toc;

                fprintf('###trial %d of %d done, time needed: %d###\n',trial,N_sims,elapsed_time)

            end

            data = [D_HR,D_LR,Prob_cue,Trial_num];

        else
            cd(data_dir)
            load(saved_data{idx_figure})
            cd(based)
        end

        D_HR      = data(:,1); 
        D_LR      = data(:,2); 
        Prob_cue  = data(:,3); 
        Trial_num = data(:,4);

        % Plot prob to choose risky as a function of trials:
        prob_cue_trialNum = splitapply(@mean,Prob_cue,Trial_num);

%         figure, set(gcf,'color','white')
        plot(prob_cue_trialNum,'-.','LineWidth',2)
        set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
        ylim([0,1.2])
        set(gca, 'XTick', [0:2:length(prob_cue_trialNum)]),xlabel('Trial Number')

    end
    
    legend({'Cue quite costly','Cue a bit costly','Cue neutral','Cue a bit rewarding','Cue quite rewarding'})
    hold off
    
end

%% ========================================================================
% 9.) Illustrate effects of other parameters - effects of d in Inference
% Illustrates effects of priors on hidden state
% Also show reversal learning experiments here
%========================================================================== 

if effects_modelparams_d
    
    %%%% USED IN PAPER %%%%
    saved_data = {'MDP_inference_Reversal' ...            % Reversal, uniform prior over context
                  'MDP_inference_Reversal_dPositive'};    % Reversal, optimistic prior over context

    simulate = false;

    N_Figures   = 1:2;
    betas       = [2^0 2^0];
    alphas      = [16 16];
    etas        = [0.5 0.5];
    ambiguities = [true true];

    n_trials    = 32;

    for idx_figure = 1:length(N_Figures)

        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        ambiguity = ambiguities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_inference(beta,alpha,eta,ambiguity);
        
        if idx_figure==2
            mdp.d{1}(1)=8;         
        end  

        if simulate
            
            MDP = mdp;

            [MDP(1:n_trials)] = deal(mdp);

            % include reversal
            [MDP(n_trials/2+1:end).s] = deal(2);

            % solve and show behaviour over trials (and epistemic learning
            MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

        else

            load(saved_data{idx_figure})

        end

        % plot
        figure(N_Figures(idx_figure)), set(gcf,'color','white')
        if idx_figure==1
            Plot_Experiment_Ambiguity(MDP,mdp);
        elseif idx_figure
            Plot_Experiment_Ambiguity_ParamsEffects(MDP,mdp);     
        end

    end
    
    %%%% USED IN PAPER %%%%
    saved_data = {'data_sim_state_ambiguity_reversal' ...           % Reversal, uniform prior over context
                  'data_sim_state_ambiguity_reversal_dPositive'};   % Reversal, optimistic prior over context

    simulate = false;

    N_Figures   = 1:2;
    betas       = [2^0 2^0];
    alphas      = [16 16];
    etas        = [0.5 0.5];
    ambiguities = [true true];

    n_trials    = 32;

    N_sims = 1000;
    
    figure, set(gcf,'color','white'), title('Probability to sample the cue first as a function of time')
    hold on    

    for idx_figure = 1:length(N_Figures)

        beta      = betas(idx_figure);        % precision of policy selection
        alpha     = alphas(idx_figure);       % precision of action selection
        eta       = etas(idx_figure);         % Learning rate
        ambiguity = ambiguities(idx_figure);  % goal-directed exploration?

        mdp       = gen_mdp_inference(beta,alpha,eta,ambiguity);
        
        if idx_figure==2
            mdp.d{1}(1)=8;        
        end          

        if simulate

            Prob_cue = [];
            Trial_num = [];
            D_HR = [];
            D_LR = [];

            for trial = 1:N_sims

                elapsed_time = 0;
                tic

                MDP = mdp;

                [MDP(1:n_trials)] = deal(mdp);

                % include reversal
                [MDP(n_trials/2+1:end).s] = deal(2);

                % solve and show behaviour over trials (and epistemic learning
                MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

                choice_prob = extractfield(MDP, 'P'); 
                choice_prob = reshape(choice_prob,4,length(choice_prob)/4); 
                choice_prob(:,2:2:n_trials*2) = [];

                prob_cue = choice_prob(4,:);

                Prob_cue = [Prob_cue;prob_cue'];

                d = extractfield(MDP, 'd');
                d_HR = zeros(1,n_trials);
                d_LR = zeros(1,n_trials);

                for i = 1:n_trials
                    d_HR(i) = d{i}{1}(1);
                    d_LR(i) = d{i}{1}(2);
                end

                D_HR = [D_HR; d_HR'];
                D_LR = [D_LR; d_LR'];

                Trial_num = [Trial_num; [1:n_trials]'];

                elapsed_time = toc;

                fprintf('###trial %d of %d done, time needed: %d###\n',trial,N_sims,elapsed_time)

            end

            data = [D_HR,D_LR,Prob_cue,Trial_num];

        else

            load(saved_data{idx_figure})

        end

        D_HR = data(:,1); 
        D_LR = data(:,2); 
        Prob_cue = data(:,3); 
        Trial_num = data(:,4);

        % Plot prob to choose risky as a function of trials:
        prob_cue_trialNum = splitapply(@mean,Prob_cue,Trial_num);

%         figure, set(gcf,'color','white'), title('Probability to sample the cue first as a function of time')
        plot(prob_cue_trialNum,'-.','LineWidth',2)
        set(gca, 'YTick', [0:0.25:1]),ylabel('Probability')
        ylim([0,1.2])
        set(gca, 'XTick', [0:2:length(prob_cue_trialNum)]),xlabel('Trial Number')
        
        if idx_figure==1
            save('data_sim_state_ambiguity_reversal','data')
        elseif idx_figure==2
            save('data_sim_state_ambiguity_reversal_dPositive','data')
        end

    end    
    
    legend({'Uniform prior','Optimistic prior'})
    hold off    
    
end

%% ========================================================================
% Sigmoid wrt information gain vs. randomness
% Figs 21
% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing
%==========================================================================

if active_learning_sigmoid
    
    %%%% USED IN PAPER %%%%
    saved_data = {'Prob_risky_21_a25' ...   % Prob choose risky for concentration param = 0.25
                  'Prob_risky_21_a5' ...    % Prob choose risky for concentration param = 0.5
                  'Prob_risky_21_a75' ...   % Prob choose risky for concentration param = 0.75
                  'Prob_risky_21_a1' ...    % Prob choose risky for concentration param = 1
                  'Prob_risky_21_noC'};    	% Prob choose risky no curiosity

    simulate = false;    

    N_Figures   = 1:5;
    Rprobs      = [0.5];
    betas       = [2^0];
    alphas      = [4];
    etas        = [0.5];
    curiosities = [true true true true false];
    
    c_high = linspace(-4,8,61);
    
    Prob_risky = nan(1,length(c_high));
    
    figure, set(gcf,'color','white'), hold on

    for idx_figure = 1:length(N_Figures)    
        
        if simulate        

            for idx_c_high = 1:length(c_high)

                Rprob     = Rprobs(1);       % probability for receiving reward in risky option
                beta      = betas(1);        % precision of policy selection
                alpha     = alphas(1);       % precision of action selection
                eta       = etas(1);         % Learning rate
                curiosity = curiosities(idx_figure);  % goal-directed exploration?

                mdp       = gen_mdp_learning(Rprob,beta,alpha,eta,curiosity);
                
                if idx_figure==2
                    mdp.a{2}(3:4,3) = 0.5;
                elseif idx_figure==3
                    mdp.a{2}(3:4,3) = 0.75;
                elseif idx_figure==4
                    mdp.a{2}(3:4,3) = 1;                    
                end
                
                mdp.C{2}(3) = c_high(idx_c_high);
                mdp.C{2}(4) = 0;

                MDP = mdp;

                % solve and show behaviour over trials (and epistemic learning
                MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

                Prob_risky(idx_c_high) = MDP.P(3);

            end
        
        else

            load(saved_data{idx_figure})

        end        

        plot(0.5*c_high-2,Prob_risky)
        title('Goal-directed exploration (active learning)')
        ylim([-0.2,1.2])
        set(gca, 'XTick', [min(0.5*c_high-2):0.5:max(0.5*c_high-2)])
        ylabel('Probability to choose informative option')
        xlabel('Expected value difference')  
    
    end
    
    legend({'CP = 0.25','CP = 0.5','CP = 0.75', 'CP = 1','No active learning'})
    hold off
    
    %%%% USED IN PAPER %%%%
    saved_data = {'Prob_risky_beta2^-3' ...  % Prob choose risky for beta = 2^-3
                  'Prob_risky_beta2^-1' ...  % Prob choose risky for beta = 2^-1
                  'Prob_risky_beta2^0' ...   % Prob choose risky for beta = 2^0
                  'Prob_risky_beta2^1' ...   % Prob choose risky for beta = 2^1
                  'Prob_risky_beta2^3'};     % Prob choose risky for beta = 2^3

    simulate = false;    

    N_Figures   = 1:5;
    Rprobs      = [0.5];
    betas       = [2^-3 2^-1 2^0 2^1 2^3];
    alphas      = [4];
    etas        = [0.5];
    curiosities = [false];
    
    c_high = linspace(-4,8,61);
    
    Prob_risky = nan(1,length(c_high));
    
    figure(212), set(gcf,'color','white'), hold on

    for idx_figure = 1:length(N_Figures)    
        
        if simulate        

            for idx_c_high = 1:length(c_high)

                Rprob     = Rprobs(1);       % probability for receiving reward in risky option
                beta      = betas(idx_figure);        % precision of policy selection
                alpha     = alphas(1);       % precision of action selection
                eta       = etas(1);         % Learning rate
                curiosity = curiosities(1);  % goal-directed exploration?

                mdp       = gen_mdp_learning(Rprob,beta,alpha,eta,curiosity);
                
                mdp.C{2}(3) = c_high(idx_c_high);
                mdp.C{2}(4) = 0;

                MDP = mdp;

                % solve and show behaviour over trials (and epistemic learning
                MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

                Prob_risky(idx_c_high) = MDP.P(3);

            end
        
        else

            load(saved_data{idx_figure})

        end        

        plot(0.5*c_high-2,Prob_risky)
        title('Random exploration (precision)')
        ylim([-0.2,1.2])
        set(gca, 'XTick', [min(0.5*c_high-2):0.5:max(0.5*c_high-2)])
        ylabel('Probability to choose informative option')
        xlabel('Expected value difference')  
    
    end
    
    legend({'beta = 2^{-3}','beta = 2^{-1}','beta = 2^0', 'beta = 2^1','beta = 2^3'})
    hold off    

end

%% ========================================================================
% Active Learning, non-linear effect of value?
% Problem: Rat in a maze has to decide whether to go for safe option (left)
% or try risky option (right) with higher reward but possibility of winning
% nothing
%==========================================================================

if active_learning_value
    
    %%%% USED IN PAPER %%%%
    saved_data = {'Prob_risky_22_flatBeta' ...    % Choice probabilities for constant beta of 2^0
                  'Prob_risky_22_ChangeBeta'};    % Choice probabilities for parametrically varying beta between 2^3 and 2^-3

    simulate = true;    

    N_Figures   = 1:2;
    Rprobs      = [0.5];
    betas       = [linspace(2^0,2^0,21);
                   linspace(2^1,2^-1,21)];
    alphas      = [4];
    etas        = [0.5];
    curiosities = [true];
    
    c_low  = linspace(0,10,21);
    c_high = linspace(0,10,21);
    
    Prob_risky = nan(length(c_low),length(c_high));

    for idx_figure = 1:length(N_Figures)    
        
        if simulate        

            for idx_c_low = 1:length(c_low)
                for idx_c_high = 1:length(c_high)

                    Rprob     = Rprobs(1);       % probability for receiving reward in risky option
                    beta      = mean(betas(idx_figure,idx_c_low)+betas(idx_figure,idx_c_high));        % precision of policy selection
                    alpha     = alphas(1);       % precision of action selection
                    eta       = etas(1);         % Learning rate
                    curiosity = curiosities(1);  % goal-directed exploration?

                    mdp       = gen_mdp_learning_ValueEffect(Rprob,beta,alpha,eta,curiosity);

                    mdp.C{2}(2) = c_low(idx_c_low);
                    mdp.C{2}(4) = c_high(idx_c_high);

                    MDP = mdp;

                    % solve and show behaviour over trials (and epistemic learning
                    MDP  = spm_MDP_VB_X_EpistemicLearning(MDP);

                    Prob_risky((length(c_low)+1)-idx_c_low,idx_c_high) = MDP.P(3);

                end
            end
        
        else

            load(saved_data{idx_figure})

        end        

        figure(N_Figures(idx_figure)), set(gcf,'color','white')
        imagesc(Prob_risky)
        if idx_figure==1
            title('Value of information as function of reward - without salience (precision)')
        elseif idx_figure==2
            title('Value of information as function of reward - with salience (precision)')
        end        
        ylabel('Reward uninformative option')
        xlabel('Reward informative option')
        set(gca, 'YTick', [1:5:length(c_low)]),set(gca, 'YTickLabel', {[c_low(length(c_low):-5:1)]})
        set(gca, 'XTick', [1:5:length(c_high)]),set(gca, 'XTickLabel', {[c_high(1:5:length(c_high))]})
        line([1 length(c_high)],[length(c_low) 1],'Color','black','LineStyle','--')
        colorbar,caxis([0 1]),colormap(hot) 
    
    end

end
    