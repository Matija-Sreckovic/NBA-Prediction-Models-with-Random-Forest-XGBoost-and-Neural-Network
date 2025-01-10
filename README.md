To run the notebook, simply download the "games_almost_ready_for_training.csv" file and run the cells of "Random Forest, XGBoost, Neural Network with CV, PCA.ipynb" in order, until the last markdown cell.

# Comments on Model Performance

## A Heads-Up on the games DataFrame

Let's explain the relevant columns in the "games_almost_ready_for_training.csv" file:
- 2 team name columns: 'team_name_home', 'team_name_away'. Note that at some point in time, the Charlotte Bobcats changed their name to the Charlotte Hornets and the New Orleans Hornets became the New Orleans Pelicans. I rename them for consistency. These will then be OneHotEncoded.
- 1 date column: 'game_date'. Since many models can't process timeframes, we set the earliest date in the DF to 0 and the rest are transformed into the number of days since the earliest date.
- 6 win-loss % columns: 'record_home_wins', 'record_home_losses', 'record_away_wins', 'record_away_losses', 'home_wl%', 'away_wl%'.
- 2 Elo columns: 'elo_home', 'elo_away'. These are Elos *before* the game. For more details on these ratings, see Step 3 of the "Data Collection for RF, XGB, NN" notebook or the ReadMe in https://github.com/Matija-Sreckovic/NBA-Simple-Prediction-Models.
- 2 team OffRtg-DefRtg-GmSc ratings columns: 'rating_home', 'rating_away'. For more details on these scores, see Step 5 of "Data Collection for RF, XGB, NN" notebook or the ReadMe in https://github.com/Matija-Sreckovic/NBA-Prediciton-Model.
- 12 individual OffRtg-DefRtg-GmSc ratings columns: 'rating_home_player1', ..., 'rating_home_player5', 'rating_away_player1', ..., 'rating_away_player5'. 5 for each of the teams' best players' ratings.
- 8 Offensive/Defensive Rating columns: 'ortg_home', 'ortg_away', 'drtg_home', 'drtg_away', 'home_ortg_last_season', ... Contains teams' offensive/defensive ratings up to that point in the season, or for all of last season.

## Conclusion for the Random Forest Model

The best performing random forest model is *without* PCA, with *no* "individual" columns removed (and differences between the teams' scores added). The best scoring parameters are n_estimators = 50, min_samples_split = 2, min_samples_leaf = 10, max_features = 'sqrt', max_leaf_nodes = 1000. Average accuracy of that model on the randomly selected cross-validation split (3-fold): 66.7%.  

- First train-test split: train from 2013-14 to 2021-22 season, test on 2022-23 and 2023-24 seasons. Best result on this split: no PCA, no column dropping, n_estimators = 50, min_samples_split = 2, min_samples_leaf = 10, max_features = None, max_leaf_nodes = None. Accuracy: 65.9%
- Second train-test split: train from 2013-14 to 2022-23 season, test on 2023-24 and 2024-25 up to Dec. 16. Best result on this split: with PCA, no column dropping, n_estimators = 50, min_samples_split = 100, min_samples_leaf = 1, max_features = None, max_leaf_nodes = 100. Accuracy: 67.1%
- Third train-test split: train from 2013-14 to 2023-24 season, test on 2024-25 up to Dec. 16. Best result on this split: with PCA, no column dropping, n_estimators = 50, min_samples_split = 100, min_samples_leaf = 1, max_features = None, max_leaf_nodes = 100. Accuracy: 69.4%. Larger accuracy here probably due to very small testing set.

The conclusion is that reducing the number of components from ~90 to 30 probably **neither increases nor decreases the model's performance.** I was wrong about dropping columns; **the best results are obtained with the maximal number of columns.**
Now we move on to the XGBClassifier! We use the same train-test splits in order to compare it to RandomForestClassifier.

## XGBoost Performance and Conclusion

The best performing model, on average, was the one with PCA and all individual columns dropped (meaning we only kept the differences between the scores, not the individual scores themselves). Accuracy: 66.5%. The train-test split is the same as the one used for random forest.

- Best performance on the first train-test split: no PCA, no columns dropped; accuracy 65.9%.
- Second: no PCA, no columns dropped; accuracy 66.6%
- Third: PCA, some columns dropped (kept the individual net ratings and individual player "ODG" ratings); accuracy 70.7%. High accuracy again probably due to very small test set.

### RF-XGB Comparison

The performances of the Random Forest model and the XGBoost model are very similar. The best average accuracy is slightly better with RF; on large test sets, RF performs slightly better. On a small test set, XGB performed significantly better.

## Neural Network Performance

On average, the neural network model performed best without PCA and with all individual columns dropped. The best parameters were the following: learning_rate = 0.001, batch_size = 256, 3 hidden layers with 128/64/32 neurons, 'relu' activation function. Average accuracy: 66.4%.

- Best performance on the first train-test split: with PCA, all individual columns dropped. learning_rate = 0.01, batch_size = 32, 5 hidden layers with 256/128/64/32/16 neurons, 'relu' activation function. Accuracy: 65.6%
- Best performance on the second train-test split: with PCA, most individual columnns dropped. learning_rate = 0.001, batch_size = 32, 3 hidden layers with 256/128/64 neurons, 'relu' activation function. Accuracy: 66.6%
- Best performance on the third train-test split: joint between:
   1) without PCA, no individual columns dropped. learning_rate = 0.01, batch_size = 32, 3 hidden layers with 128/64/32 neurons, 'relu' activation function, and
   2) without PCA, most individual columns dropped. learning_rate = 0.001, batch_size = 256, 3 hidden layers with 128/64/32 neurons, 'relu' acrivation function.
Accuracy of both: 68.6%

The neural network model performed slightly worse than the RF and XGB models. This could be due to the fact that the parameter grid for tuning had less elements than for the RF and XGB models, which is itself due tot he fact that the NN model takes longer to train.

## Overall Conclusion

All three models displayed very similar performance, with RF and XGB slightly ahead of NN. Hyperparameter tuning did not cause dramatic improvement. The best performance was by far on the third train-test split, which is possibly due to the very small test set (389 games compared to 1500-2000 in the first two), which causes higher variance of accuracy w/r/t change in hyperparameter values.

My main hope is that the next jump in model performance could be achieved by measuring teams' scores only on their last 10/20/30 games. Although a decent indicator of a player's quality, performance over a full season, and especially over the last 2/3/5 seasons is probably too long a period to make near-future predictions.

## Explanation of feature engineering:
See the markdown cells of "Data Collection for RF, XGB, NN.ipynb".
