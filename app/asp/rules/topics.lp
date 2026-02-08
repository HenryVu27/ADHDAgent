% ============================================
% Topic Boundary Definitions
% ============================================
% Defines the structured topic hierarchy for ADHD coaching conversations.
% Each topic has sub-topics and associated evidence-based strategies.

% --- Top-Level Topics ---
topic(daily_routines).
topic(homework).
topic(emotional_regulation).
topic(social_skills).
topic(transitions).
topic(positive_reinforcement).
topic(parent_self_care).

% --- Daily Routines Sub-Topics ---
subtopic(daily_routines, morning_routine).
subtopic(daily_routines, bedtime_routine).
subtopic(daily_routines, meal_time).
subtopic(daily_routines, chore_completion).

% --- Homework Sub-Topics ---
subtopic(homework, getting_started).
subtopic(homework, staying_focused).
subtopic(homework, completing_assignments).
subtopic(homework, organization).

% --- Emotional Regulation Sub-Topics ---
subtopic(emotional_regulation, frustration).
subtopic(emotional_regulation, anger).
subtopic(emotional_regulation, anxiety).
subtopic(emotional_regulation, meltdowns).

% --- Transitions Sub-Topics ---
subtopic(transitions, activity_switching).
subtopic(transitions, leaving_house).
subtopic(transitions, ending_screen_time).
subtopic(transitions, new_environments).

% --- Strategy-Topic Mappings ---
% Which strategies apply to which topics
strategy_for(daily_routines, visual_schedule).
strategy_for(daily_routines, first_then_board).
strategy_for(homework, timer_technique).
strategy_for(homework, chunking).
strategy_for(homework, body_doubling).
strategy_for(emotional_regulation, calm_down_corner).
strategy_for(emotional_regulation, emotion_naming).
strategy_for(transitions, countdown_warnings).
strategy_for(transitions, transition_object).
strategy_for(positive_reinforcement, token_economy).
strategy_for(positive_reinforcement, specific_praise).

% --- Topic Relevance Detection ---
% Maps predicate categories to topics
maps_to_topic(distraction, homework).
maps_to_topic(avoidance, homework).
maps_to_topic(meltdown, emotional_regulation).
maps_to_topic(tantrum, emotional_regulation).
maps_to_topic(wont_stop, transitions).
maps_to_topic(routine, daily_routines).
maps_to_topic(morning, daily_routines).
maps_to_topic(bedtime, daily_routines).
