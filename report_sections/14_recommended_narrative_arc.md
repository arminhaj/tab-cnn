# Recommended Narrative Arc

The report reads most clearly if it begins by framing tablature estimation as a stricter and more instrument-aware task than generic pitch transcription. From there, it can present the CNN as a strong baseline for local time-frequency pattern recognition and motivate the CRNN as a targeted extension that adds short-range temporal modeling.

The middle of the report should then follow the actual project path: modernization of the original code, early collapse toward the mute class, diagnosis of that failure, a stronger but larger bidirectional-GRU CRNN phase, and finally the stabilized comparison against a smaller unidirectional GRU. That sequence makes the empirical argument much stronger because it shows why the final comparison is trustworthy.

The ending should show that recurrence helps on pitch-level recovery but does not automatically improve exact fingering prediction. That sets up the final message: model quality in this domain depends both on architecture and on careful training behavior, and the hardest remaining challenge is the ambiguity of guitar fingering rather than simply the absence of temporal context.
