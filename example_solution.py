# This example uses Python 3.12.9, pandas, and scikit-learn

import ast
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LogisticRegression 

# ── Load Data ──────────────────────────────────────────────────────────────────

tracks = pd.read_csv("data/tracks.csv")
users_train = pd.read_csv("data/users_train.csv")
users_holdout = pd.read_csv("data/users_holdout.csv")
#sessions_train = pd.read_csv("data/streaming_sessions_train.csv")
sessions_holdout = pd.read_csv("data/streaming_sessions_holdout.csv")

print(tracks.head())
print(users_train.head())
print(sessions_holdout.head())

# ── Task 1: Skip Prediction ────────────────────────────────────────────────────
# For each session in the holdout set, predict whether the last song was skipped.
# The last entry in skip_labels is null — that's what we're predicting.
#
# This baseline: look up the last track's historical skip rate from tracks.
# If more than 50% of its plays were skips, predict 1 (skip), otherwise 0 (play).
# You can do much better — try using the session's skip history as context,
# or incorporating user-level features.

# Build a lookup: track_id -> avg_skip_pct
skip_rate = tracks.set_index("track_id")["avg_skip_pct"].to_dict()

skip_predictions = []
for _, row in sessions_holdout.iterrows():
    track_ids = ast.literal_eval(row["track_ids"])   # stored as a list string in the CSV
    last_track_id = track_ids[-1]
    rate = skip_rate.get(last_track_id, 0.0)         # default to 0 if track is unknown
    skip_predictions.append(1 if rate > 0.5 else 0)

# Write Task 1 submission
sessions_submission = pd.read_csv("submission/streaming_sessions_submission.csv")
sessions_submission["predicted_skip_label(0/1)"] = skip_predictions
sessions_submission.to_csv("submission/streaming_sessions_submission.csv", index=False)
print(f"Task 1: predicted {sum(skip_predictions)} skips out of {len(skip_predictions)} sessions")

# ── Task 2: Promotion Allocation ───────────────────────────────────────────────
# 500 users are about to leave. We can promote exactly 125 of them.
# Our score is the total future CLV of the 125 users we pick.
#
# This baseline: train a random forest to predict future_clv, then promote
# the 125 users with the highest predicted value.
# You can do much better — try feature engineering or other regression models.

FEATURES = ["avg_skip_pct", "total_profit", "age", "play_count", "unique_song_count", "liked_song_count"]
N_PROMOTIONS = 125

# Train on users where we know the CLV
X_train = users_train[FEATURES].fillna(0)
y_train = users_train["future_clv"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict CLV for holdout users and pick the top 125
X_holdout = users_holdout[FEATURES].fillna(0)
users_holdout["predicted_clv"] = model.predict(X_holdout)
top_ids = set(users_holdout.nlargest(N_PROMOTIONS, "predicted_clv")["user_id"])

# Write Task 2 submission
users_submission = pd.read_csv("submission/users_submission.csv")
users_submission["selected(0/1)"] = users_submission["user_id"].apply(lambda uid: 1 if uid in top_ids else 0)
users_submission.to_csv("submission/users_submission.csv", index=False)
print(f"Task 2: selected {users_submission['selected(0/1)'].sum()} users for promotion")