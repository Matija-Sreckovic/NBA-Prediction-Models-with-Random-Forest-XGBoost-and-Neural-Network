To run the notebook, simply download the "games_almost_ready_for_training.csv" file and run the cells of "Random Forest, XGBoost, Neural Network with CV, PCA.ipynb" in order, until the last markdown cell. You can get a different cross-validation train-test split using the "get_cross_val_indices" function. If you want the one I used, run the cell with the following code:
```python
training_indices = [0, 0, 0]
testing_indices = [0, 0, 0]

training_indices[0] = list(range(11503))
testing_indices[0] = list(range(11503, 14136))

training_indices[1] = list(range(12187))
testing_indices[1] = list(range(12188, 14525))

training_indices[2] = list(range(14136))
testing_indices[2] = list(range(14136, 14525))

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

## Table of models' performance

| Model Name | Avg Accuracy | 1st train-test split | 2nd train-test split | 3rd train-test split |
|----------|----------|----------|----------|----------|
|   Random Forest  |   66.7% |   65.9%  |   67.1%  |   69.4%  |
|   XGBoost  |   66.5%  |   65.9%  |   66.6%  |   70.7%  |
|   Neural Network  |   66.4%  |   65.6%  |   66.6%  |   68.6%  |
|   ODG rating + home-court advantage  |   66.2%  |   65.9%  |   66.8%  |   65.8%  |

# Explanation of feature engineering:

## Elo Ratings

These are margin-adjusted Elo ratings of each team as columns of games_new. For this, we start from the first NBA game ever played in 1947! But we only care about the 2013-14 season and onwards.

Each team starts with a rating of 1200. The home team has a 114 Elo advantage, and teams playing the second night of a back-to-back have their Elo reduced by around 20 (there is a small variance depending on whether the back-to-backs were played at home or away). 

I developed this in a previous project: https://github.com/Matija-Sreckovic/NBA-Simple-Prediction-Models

The formula for the new Elo rating after the game is:

$$E_{new} = E_{old} + P \times G \times (result - e_r),$$ where:

- $P = 15$ for regular season games, and $P = 10$ for playoff games
- $G = (1 + N_{game}/N_{median})^{1/3}$, where $N_{game}$ is the game's margin and $N_{median}$ the median margin of victory this season; or $G = 1$ if it's the first game of the season.
- $result = 1$ if the team won, $0$ if the team lost 
- $e_r = q_{team} / (q_{team} + q_{opponent})$, where $q_{team} = 10^{Elo_{team}/400}, q_{opponent} = 10^{Elo_{opponent}/400}$ (stands for "expected result")

I found this formula at https://www.aussportstipping.com/sports/nba/elo_ratings/. I tried to tune the $1/3$ parameter but this was more or less the best value.

## OffRtg-DefRtg-Gamescore ("ODG") Ratings

### GameScore, Off/DefRtg

I'll try to explain how the rating system works here. It uses two catch-all advanced stats of a box score: a player's **GameScore** (**GmSc**) and the difference between his **OffRtg** and **DefRtg**. The formula for **GmSc** is:

$$ GmSc = PTS + 0.4 \times FG - 0.7 \times FGA - 0.4\times(FTA - FT) + 0.7 \times ORB + 0.3 \times DRB + STL + 0.7 \times AST + 0.7 \times BLK - 0.4 \times PF - TOV.$$

On the other hand, **OffRtg** and **DefRtg** for a *team* are the number of points a team scores/allows per 100 possessions. For a *player*, the formula is much more complicated (see https://www.basketball-reference.com/about/ratings.html), but according to its creator, Dean Oliver, the author of ["Basketball on Paper"](https://www.amazon.com/Basketball-Paper-Rules-Performance-Analysis/dp/1574886886):

**OffRtg:** "Individual offensive rating is the number of points produced by a player per hundred total individual possessions. In other words, 'How many points is a player likely to generate when he tries?'"

**DefRtg:** "The core of the Defensive Rating calculation is the concept of the individual Defensive Stop. Stops take into account the instances of a player ending an opposing possession that are tracked in the boxscore (blocks, steals, and defensive rebounds), in addition to an estimate for the number of forced turnovers and forced misses by the player which aren't captured by steals and blocks."

### The Rating System Itself - 1 Game

A **score** is assigned to each player after each game he played in.

We take in each player's GmSc and difference between OffRtg and DefRtg (henceforth "Rtg"). We compare them to all other scores of players in all of the last 5 years - for example, if each is 99th percentile, the player gets a coefficient of 0.99 for GmSc and Rtg.

To get the player's *unweighted rating*, we set

$$
\textup{unweighted rating} = 0.2 \times \textup{GmSc coefficient} + 0.8 \times \textup{Rtg coefficient}.
$$

The value $0.2$ was tuned.

To get the player's *weighted rating* we multiply the unweighted rating by a **usage rate coefficient** (we get the player's USG% and assign a coefficient by comparing it to all other games in the last 5 years, similarly to how we obtain the Gmsc and Rtg coefficients) and **minutes_coefficient** (if a player played at least 35 minutes, the coefficient is 1). Precisely, 

$\textup{minutes coefficient} = 6.85 \times \frac{\textup{player's minutes played}}{\textup{total team minutes}}.$ 

In total, 

$$\textup{weighted rating} = \textup{USG% coefficient} \times \textup{minutes coefficient} \times (0.2 \times \textup{GmSc coefficient} + 0.8 \times \textup{Rtg coefficient}).$$

### Players' Long-Term Ratings and a Team's Rating before a Game

The score that we compute for a player after each game is added to his rating sum *this season* and his *5-year rating sum*. Both are divided by the number of games played to get the *average rating this season* and *average 5-year rating*. There are also *last season's average ratings* and *average ratings from 2 seasons ago*.

To compute a player's rating before a game, we use a weighted average of the 4 parameters above (this season's avg rating so far, last season's avg rating, avg rating from 2 seasons ago, 5-year avg rating). The weights depend on how many games the player has played this season and the previous 2 seasons. For example, if a player has played 15 games this year, 55 games last year and 0 games 2 years ago, the formula would be:

$$\textup{player game rating} = 0.5 \times \textup{rating this season} + 0.3 \times \textup{rating last season} + 0 \times \textup{rating 2 seasons ago} + 0.2 \times \textup{rating in the last 5 years}.$$

The weights were selected a bit arbitrarily, but tuning did not improve the model's performance.

At the end of this process, we get a vector of ratings for about 10-15 players; we take all **uninjured** players for the game! Call this vector $v_{\textup{ratings}}.$ The team's score is the dot product of this vector with the **vector of players' usage estimates**.

The players' usage estimates are numbers that take into account each player's average usage rate and minutes played. For example, if a player's average usage rate is 20% this season, 30% last season, and 27% in the last 5 years, and the player played 80 games both this season and the last, then

$$ \textup{player game usage coeff} = 0.6 \times \textup{this year's USG\%} + 0.2 \times \textup{last year's USG\%} + 0.2 \times \textup{5 year USG\%}. $$

There is also a minutes played usage coefficient, computed as follows: we take the player's average minutes coefficient computed earlier for both this season and last season (so if a player played at least 35 minutes in each game he played, both are $1$), compute a weight $\textup{minutes weight} =  \min(1, \textup{games played this season}  \times 0.1)$, and compute:

$$\textup{player game minutes coeff} = (\textup{minutes weight}) \times (\textup{avg minutes coeff this season}) + (1 - \textup{minutes weight}) \times (\textup{avg minutes coeff last season}).$$

Then we get each player's overall weight for the game by multiplying these:

$$\textup{player game weight} = \textup{player games usage coeff} \times \textup{player game minutes coeff}.$$

We take these game weights together into a vector $v_{\textup{weights, unnormalized}}$, normalize it:

$$v_{\textup{weights}} = \frac{v_{\textup{weights, unnormalized}}} {||v_{\textup{weights, unnormalized}}||_2},$$

where the $2$-norm was chosen because it worked best (the initial idea was the $1$-norm; I tuned this value too). Finally, we take the dot product of $v_{\textup{ratings}}$ and $v_{\textup{weights}}$:

$$\textup{team score} = \langle v_{\textup{ratings}}, v_{\textup{weights}} \rangle.$$

Usually, the best players' ratings are around 40, with quick fall-offs towards the 20's. Negative ratings are not uncommon! The prediction is that the team with the greater score wins. There should also be a home-court advantage factor - the one that worked best for the 2023-24 season is +3 for the home team.
