# Predicting Chess Match Outcomes Using Machine Learning

## Problem/Motivation Statement

The game of chess has been played for thousands of years. Since the very first chess
game, strategy and tactics in the game have been constantly evolving. Even now, in an age
where a chess bot hosted on a cell phone can beat the greatest humans in the world, we are
still discovering new approaches to the game.

One area of chess that has remained relatively innovation-free is the rating system,
known as Elo. This system was invented by a master-level chess player and physics professor
named Arpad Elo in the 1950s. In 1970 it was adopted as the official ratings system by the
largest chess organization in the world (FIDE) and has remained unchanged ever since.

In the Elo system, chess players are assigned numerical ratings typically in the
1000-2000 range. These ratings are meant to be a representation of a player’s overall strength
at the game. When two players compete, the expected outcome of the game can be computed
by comparing their Elo scores. For example, with a rating difference of 200 points, the
higher-rated player would be expected to win about 75% of the time. After each game, players’
ratings are adjusted up or down depending on whether they won or lost the game.

Scholars have pointed out a number of issues with the Elo system. Arpad Elo devised a
number of simplifying assumptions to his model to make it easier to calculate. For example, the
system assumes that the standard deviation of all players’ ability for a given chess game is
equal. In reality, players most likely all have different levels of variation in their performance on a
game-by-game basis. Such assumptions were necessary in 1970 due to a lack of computing
power.

In 2023, we believe that we can improve on the standard Elo system for estimating the
outcome of chess games. We aim to incorporate Elo as well as other features of a chess game
like the time limit into a machine learning model that will surpass the predictive ability of Elo.


## Dataset and Analytic Goals

We pulled our data from two sources: Chess.com and Lichess.org. These are the largest
and second-largest chess platforms in the world, respectively. They are also incredibly rich and
accessible data sources that have more information than we could possibly use. Lichess.org for
example allows you to download the monthly archives of all games played on the platform.
These archives are about 60GB per month.

We used the Chess.com API data to construct our game outcome prediction model. We
pulled game-level data from some of the top grandmasters on the platform to build a reliable
base dataset to train our model. We included the following variables: _White Elo_ - the elo of the
player with the white pieces _, Black Elo_ - the eloof the player with the black pieces _, Result_ -
which player won the game(1 - 0 : white, 0 - 1 : black), _Time Class_ - what type of game was
played _, Time Control_ - the amount of minutes eachplayer has in the game, _Opening_ - the name
of the opening that was played, _Date_, _Start Time_, and _End Time_. We knew that _White Elo_ and
_Black Elo_ would be the most important feature forpredicting _Result_, but we included the other
variables to experiment with further improving the model.

## Data Engineering Pipeline

This data pipeline uses Python scripts to fetch data from Lichess and Chess.com APIs
and store them in Google Cloud Platform (GCP) buckets. Then, Spark is used to process the
data from GCP and store it in a MongoDB database. The entire process is orchestrated using
Airflow, which provides a platform for managing the various steps of the pipeline and scheduling
the execution of the scripts. To accomplish our machine learning goals, we utilize a Databricks
cluster where we retrieve the data stored in MongoDB and employ SparkMLib for modeling.


## Preprocessing

Some of the preprocessing for the Chess.com data is done before we upload the data to
GCP, and some is done after it has been uploaded to our MongoDB instance.

There were a few issues we had to resolve while preprocessing the API data. The first
problem was that the “result” column had many unexpected values. This is because there are
actually a multitude of different ways to win or lose in a game of chess. For example, you can
lose by running out of time, getting checkmated, or just resigning. Each of these possibilities
had its own code associated with it. We encoded the twelve or so different result codes into just
three: win, lose, and draw. Later in the pipeline we decided to drop games that ended in draws
since Elo doesn’t effectively predict the probability of a draw.

Another problem was that the output data from the API was not standardized, so we
couldn’t assume that, for example, index 5 of the _PGN_ list always held the same type of data. To
address this, we added error handling so that only games of a particular form would be added to
our training data.

Once our data was in MongoDB, we did a variety of preprocessing to create better
features for our models. We created a new interaction feature ‘elo_diff’ which was the difference
between the elo rating of the player with white pieces and the player with black pieces. The
_Time Class_ and _Time Control_ variables were representedas strings, so we converted them to
be categorical using SparkML’s StringIndexer and OneHotEncoder functions. This reformatting
took about 2 minutes to execute on our cluster.

Our cluster configuration was as follows:

- Databricks Runtime Version: 7.3 LTS (includes Apache Spark 3.0.1, Scala 2.12)
- Workers: 61-152.5 GB Memory, 8-20 Cores
- Driver: 30.5 GB Memory, 4 Cores
- Number of Workers: 2-


Cluster configuration (with GPU):

- Databricks Runtime Version: 12.1 ML (includes Apache Spark 3.3.1, GPU, Scala 2.12)
- Workers: 32-80 GB Memory, 8-20 Cores
- Driver: 16 GB Memory, 4 Cores
- Number of Workers: 2-

## ML Goals

All the relevant features, post-processing, were transformed into a single column using
‘VectorAssembler’. The resulting dataframe was split into train and test (0.8/0.7 train and 0.2/0.
test) before training the models using SparkMLib.

Our overarching machine learning goal was to create a score prediction algorithm that
surpasses the traditional Elo in both performance and flexibility. In order to do this, we explored
a wide variety of machine learning models:

1. Decision Trees
2. Random Forests
3. Logistic Regression
4. Support Vector Machines
5. Gradient-Boosted Trees
6. Naive Bayes Classifier

We implemented these models and compared their ability to predict the outcome of a
game with that of the Elo system. Most of the models were tuned and cross validated using
‘ParamGridBuilder()’ and ‘CrossValidator()’ respectively. Finally, the model was evaluated using
‘BinaryClassificationEvaluator()’.

## Outcomes

```
Below are the metrics that we obtain with the ELO Formula (Benchmark) :-
```
```
Model Accuracy AUC PR
Elo Formula Prediction
(Benchmark) 70.03% 0.7029 0.
```

The first notable result here is that Elo actually is quite good at predicting chess
outcomes by itself. Chess can be an unpredictable and volatile game, and the quality of a player
on any given day is affected by hundreds of variables that cannot realistically be modeled.
Taking that into account, Elo attained an accuracy of 70%, AUC of .7, and a PR of .67. Not bad!
In fact, the best accuracy obtained by our massive ensemble of models was only 2.
percentage points better than Elo.

Below are the metrics that we obtain from ML models:-

```
Model Features Accuracy AUC PR
```
```
Decision Trees
```
```
"Black Elo", "White Elo","elo_diff", "Time
Class", "Time Control" 69.32% 0.6507 0.
"Black Elo", "White Elo" 66.50% 0.6253 0.
"elo_diff" 71.40% 0.6974 0.
```
```
Logistic Regression
```
```
"Black Elo", "White Elo","elo_diff", "Time
Class", "Time Control" 70.10% 0.7674 0.
"Black Elo", "White Elo" 71.58% 0.7766 0.
"elo_diff" 69.49% 0.7653 0.
```
```
Naive Bayes
```
```
"Black Elo", "White Elo","elo_diff", "Time
Class", "Time Control" 55.91% 0.5303 0.
"Black Elo", "White Elo" 72.14% 0.4911 0.
"elo_diff" 38.08% 0.5 0.
```
```
Random Forest
```
```
"Black Elo", "White Elo","elo_diff", "Time
Class", "Time Control" 70.06% 0.7427 0.
"Black Elo", "White Elo" 68.44 % 0.7505 0.
"elo_diff" 63.54% 0.7553 0.
```
```
SVM Classifier
```
```
"Black Elo", "White Elo","elo_diff", "Time
Class", "Time Control" 66.76% 0.7602 0.
"Black Elo", "White Elo" 56.00% 0.7295 0.
"elo_diff" 63.54% 0.7553 0.
```
```
Gradient Boosted Trees
```
```
"Black Elo", "White Elo","elo_diff", "Time
Class", "Time Control" 64.85% 0.7605 0.
"Black Elo", "White Elo" 70.00% 0.7687 0.
"elo_diff" 71.89% 0.7811 0.
```
The machine learning models truly start to differentiate themselves when we consider
the metrics of AUC and PR. Our best model AUC was nearly .1 better than the AUC of Elo,
which is massive for a metric that is essentially bounded between .5 and 1. It was a similar story


for PR. Effectively, this means that our models did a better job avoiding false positives/negatives
than Elo did. In this particular case, it appears that Elo overpredicted the amount of wins that the
white player would get, leading to its AUC and PR to be worse than its accuracy.

Another very interesting result is that adding additional features to our model beyond Elo
did not help their prediction quality, and in fact harmed it. We hypothesized that knowing the
time limit of a game would help add valuable information to our model. However, it appears that
this only added noise to our models.

## Runtime Efficiency

```
Machine Learning Algorithm Average Model Runtime
(Regular cluster)
```
```
Average Model Runtime
(GPU)
Decision Trees 1.24 minutes 2.34 minutes
Random Forest 38.57 minutes 29.6 minutes
Logistic Regression 1.08 minutes 28.7 seconds
Support Vector Machine 20.67 seconds 1.06 minutes
Gradient Boosting 15.8 minutes 10.67 minutes
Naive Bayes 2.46 seconds 52.24 seconds
```
Weirdly enough, we observe that, on an average, the model runtimes are faster on the
cluster with no GPU. On a model level, RandomForest, GradientBoosting and Logistic
Regression have a faster execution time on the cluster with GPU. One possible explanation for
this could be the fact that all of us were trying to test the runtime efficiency together which might
have increased the overhead for the cluster which added to the latency of the overall compute
time.

## Lessons Learned

One very important lesson we all learned is that modeling chess outcomes is hard. We
had a lot of ideas about using tons of different features in our models, but the reality is that most
variables besides past win/loss history will be useless in predicting outcomes. This is what Elo
effectively models already.
As we have demonstrated, there is room to improve upon Elo, but the opportunities are
more limited than we first expected.

## Future Work

An interesting area we didn’t have time to explore is the idea of personalization of the
prediction system. Instead of having an algorithm that takes two numbers and spits out the likely
winner, it could also take an individual player’s games into account. We could place higher
weights on the games played by that player during the fitting of our machine learning model,
potentially leading to more accurate, personalized predictions.

## Conclusion

The Elo system has served as an excellent baselinemodel for chess rankings for a long
time. However, it may be time to retire the system in favor of more powerful and accurate
machine learning methods. This finding has implications for the broader world of competitive
games, as Elo is currently also used to rank players in baseball, basketball, tennis, and many
other sports.


