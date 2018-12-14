% Auxiliary function for creating MDP model for active learning task
% Input:
% Rprob     = true reward prob in risky option
% beta      = hyper-prior on precision of policy selection (higher = less precise)
% alpha     = hyper-prior on precision of action selection  (higher = more precise)
% eta     = learning rate
% curiosity = active learning on or off
% 
% Output:
% mdp model containing observation model, transition probs etc

function mdp = gen_mdp_learning_ValueEffect(Rprob,beta,alpha,eta,curiosity)

% outcome probabilities: A
%--------------------------------------------------------------------------
% We start by specifying the probabilistic mapping from hidden states
% to outcomes;
%--------------------------------------------------------------------------

p          = Rprob;
q          = 1 - Rprob;

% Factor 1: Location, exteroceptive - no uncertainty about location
%--------------------------------------------------------------------------
% Not interesting for task
A{1} = [1 0 0;         % starting position
        0 1 0;         % option A
        0 0 1];        % option B

% Factor 2: Reward, interooceptive - uncertainty about reward prob
%--------------------------------------------------------------------------
% That's were true reward prob comes in
A{2} = [1 0 0          % reward neutral  (starting position)
        0 p 0          % high reward option A
        0 q 0          % no reward option A       
        0 0 p          % high reward option B
        0 0 q];        % no reward option B
    
% beliefs about outcome (likelihood) mapping
%--------------------------------------------------------------------------
a{1} = A{1}*128;
% That's where learning comes in - start with uniform prior
a{2} = [1 0 0          % reward neutral  (starting position)
        0 1 0          % high reward option A
        0 1 0          % no reward option A
        0 0 1/4        % high reward option B
        0 0 1/4];      % no reward option B
    
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
% Preferences over outcomes - determines choice between safe vs. risky

C{1}  = [0 0 0]'; % preference for first factor - doesn't care

cA = 2^2; % preference for reward option A (# of pellets)
cB = 2^2; % preference for reward option B (# of pellets)

C{2}  = [0 cA 0 cB 0]'; % preference for: [staying at starting point | safe | risky + reward | risky + no reward]

% Prior beliefs about initial states, in terms of counts. Here,
% the hidden states are factorised into location and context:
%--------------------------------------------------------------------------
D{1}  = [1 0 0]'; % prior over starting point - rat 'starts' at starting point (not at safe or risky option

% Allowable policies (of depth T).  These are just sequences of actions
% (with an action for each hidden factor)
%--------------------------------------------------------------------------
V     = [2 3]; % stay, go left, go right

% MDP Structure
% nothing to change here for now
%==========================================================================
mdp.V = V;                    % allowable policies
mdp.A = A;                    % observation process
mdp.a = a;                    % observation model
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % prior over initial states
mdp.s = 1;


mdp.Aname = {'exteroceptive','interoceptive'};
mdp.Bname = {'position'};

mdp.alpha      = alpha;                % precision of action selection
mdp.beta       = beta;             % precision of policy selection
mdp.curiosity  = curiosity;        % Goal-directed exploration

% Learning rate
mdp.eta   = eta;


end