% ============================================
% Conversation Transition Rules
% ============================================
% These rules define valid state transitions in the coaching conversation.
% The ASP solver computes answer sets representing valid next moves.

% --- Conversation Phases ---
phase(intake).
phase(strategy).
phase(progress).
phase(followup).

% --- Valid Transitions ---
% From intake, can move to strategy once we have enough context
transition(intake, strategy) :-
    has_child_age,
    has_challenge_area,
    has_attempted_strategies.

% From strategy, can move to progress once a strategy is selected
transition(strategy, progress) :-
    strategy_selected(S).

% Can always return to strategy from progress
transition(progress, strategy).

% Followup is available from any phase after first session
transition(Phase, followup) :-
    phase(Phase),
    session_count(N),
    N > 1.

% --- Current Phase Actions ---
% What actions are valid in each phase
valid_action(gather_info) :- current_phase(intake).
valid_action(recommend_strategy) :- current_phase(strategy).
valid_action(track_progress) :- current_phase(progress).
valid_action(check_in) :- current_phase(followup).

% --- Phase Completion Predicates ---
% These get asserted as the conversation progresses
% has_child_age.          % Asserted when parent shares child's age
% has_challenge_area.     % Asserted when parent identifies challenges
% has_attempted_strategies. % Asserted when parent shares what they've tried
