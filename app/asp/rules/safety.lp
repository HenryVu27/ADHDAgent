% ============================================
% Safety Constraint Rules
% ============================================
% These constraints MUST hold in every answer set.
% They ensure the chatbot stays within clinician-informed bounds.

% --- Hard Safety Constraints ---
% Never provide medical advice
:- valid_action(provide_medical_advice).

% Never recommend specific medications
:- valid_action(recommend_medication).

% Never diagnose conditions
:- valid_action(diagnose_condition).

% Never provide legal advice
:- valid_action(provide_legal_advice).

% --- Crisis Detection ---
% If crisis indicators are present, only safety actions are valid
valid_action(safety_redirect) :- crisis_detected.
:- valid_action(A), crisis_detected, A != safety_redirect.

% --- Topic Boundaries ---
% The chatbot may only discuss approved topic areas
approved_topic(behavioral_strategies).
approved_topic(daily_routines).
approved_topic(homework_support).
approved_topic(emotional_regulation).
approved_topic(communication_skills).
approved_topic(positive_reinforcement).
approved_topic(transition_planning).
approved_topic(self_care_for_parents).

% Cannot discuss unapproved topics
:- discusses(Topic), not approved_topic(Topic).

% --- Response Constraints ---
% Responses must be empathetic
:- response_generated, not tone(empathetic).

% Responses must be actionable (provide concrete steps)
:- response_generated, current_phase(strategy), not has_action_steps.

% Responses must acknowledge parent's feelings
:- response_generated, parent_expressed_frustration, not acknowledgment_present.
