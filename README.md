[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vN7Ym3SQ)
# DataHack2026

Official repository for DataHack 2026

## Getting Started

1. Join the [Discord Server](https://discord.gg/fUerVrUDnq) for resources, announcements, questions, etc.
2. Add your teammates to this repo (Settings -> Collaborators -> Add people)
3. Update `submission/team_info.json` with your team name, number, and members! We need this to score your submission!
4. For details on how to use GitHub/Git, click [here](https://medium.com/@lahari.kethinedi/github-a-step-by-step-guide-from-clone-to-push-ffad25b6313a)

## File Structure

- `data/` see Problem Description linked on the Discord Server for more details on the data and features
  - `README.md` instructions on how to download the csv files
  - After downloading/unzipping the data, you should have the following files:
  - `streaming_sessions_train.csv` historical listening sessions with full skip labels (training data for Task 1)
  - `streaming_sessions_holdout.csv` sessions where the last skip label is hidden (your Task 1 test set)
  - `users_train.csv` user profiles with future CLV labels (training data for Task 2)
  - `users_holdout.csv` at-risk user profiles without CLV labels (your Task 2 test set)
  - `tracks.csv` track metadata and audio features
- `submission/` see 'Submission' for more details
  - `team_info.json`
  - `streaming_sessions_submission.csv`
  - `users_submission.csv`
  - `slides.pptx`
  - *Note: please do not add/remove any files in this directory!*
- `README.md`
- `<any code you write>`
- `<any folders you add>`

## Submission

Please update, commit, ***and push*** the following files to your GitHub Classroom repo:

**Due at 3:30 pm:**

- `submission/team_info.json`: Team name, number, and members
- `submission/streaming_sessions_submission.csv`: Skip predictions (0 or 1) for each holdout session. These should correspond to the sessions in `data/streaming_sessions_holdout.csv`.
- `submission/users_submission.csv`: Promotion selections (at most 125 ones and at least 375 zeros) for holdout users where 1 indicates a promotion. These should correspond to the users in `data/users_holdout.csv`.
- **Note: PLEASE DO NOT CHANGE/ADD/REMOVE COLUMNS OR ROWS IN EITHER CSV**
- *Note: we will pull all your repos at exactly 3:30 pm, so any changes to your csv files pushed after that won't be counted towards your submission!*

**Due at 4:00 pm:**

- `submission/slides.pptx`: Presentation slides
- Code: Please submit what you used to analyze the data and generate your `submission/*.csv` files. This should be well commented/readable.
- *Note: your repo will lock at 4:00 pm, so please push your final slides/code before then!*
