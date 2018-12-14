% Auxiliary function for creating MDP model for active learning task
% Input:
% beta      = hyper-prior on precision of policy selection (higher = less precise)
% alpha     = hyper-prior on precision of action selection  (higher = more precise)
% eta       = learning rate
% curiosity = active learning on or off
% ambiguity = active inference on or off
% 
% Output:
% mdp model containing observation model, transition probs etc

function mdp = gen_mdp_inference(beta,alpha,eta,ambiguity)

% outcome probabilities: A
%--------------------------------------------------------------------------
% We start by specifying the probabilistic mapping from hidden states
% to outcomes.
%--------------------------------------------------------------------------

a = .75;
b = 1 - a;

c = .75;
d = 1 - c;

A{1}   = [1 1 0 0 0 0 0 0;    % ambiguous starting position (centre)
          0 0 1 1 0 0 0 0;    % safe arm selected and rewarded
          0 0 0 0 a d 0 0;    % risky arm selected and rewarded
          0 0 0 0 b c 0 0;    % risky arm selected and not rewarded
          0 0 0 0 0 0 1 0;    % informative cue - high reward prob
          0 0 0 0 0 0 0 1];   % informative cue - low reward prob
      
clear('a'),clear('b'),clear('c'),clear('d') 

% controlled transitions: B{u}
%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions of hidden states
% under each action or control state. Here, there are four actions taking the
% agent directly to each of the four locations.
%--------------------------------------------------------------------------
b{1}(:,:,1)  = [1 0 0 1; 0 1 0 0;0 0 1 0;0 0 0 0];     % move to/stay in the middle
b{1}(:,:,2)  = [0 0 0 0; 1 1 0 1;0 0 1 0;0 0 0 0];     % move up left to safe  (and check for reward)
b{1}(:,:,3)  = [0 0 0 0; 0 1 0 0;1 0 1 1;0 0 0 0];     % move up right to risky (and check for reward)
b{1}(:,:,4)  = [0 0 0 0; 0 1 0 0;0 0 1 0;1 0 0 1];     % move down (check cue)

for i = 1:4
    B{1}(:,:,i) = kron(b{1}(:,:,i),eye(2));
end

clear('b')

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
d{1}  = kron([1/4 0 0 0],[1 1])';

% Allowable policies (of depth T).  These are just sequences of actions.
%--------------------------------------------------------------------------
V  = [1  1  1  1  2  3  4  4  4  4
      1  2  3  4  2  3  1  2  3  4];

% MDP Structure
% nothing to change here for now
%==========================================================================
mdp.V = V;                    % allowable policies
mdp.A = A;                    % observation model
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = d;                    % prior over initial states
mdp.d = d;                    % prior over initial states
mdp.s = 1;                    % initial state - high reward context

% Learning rate
mdp.eta   = eta;
mdp.beta  = beta;
mdp.alpha = alpha;

mdp.curiosity = false;
mdp.ambiguity = ambiguity;

end