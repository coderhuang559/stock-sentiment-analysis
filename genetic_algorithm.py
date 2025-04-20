def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from genetic_operations import mutation,crossover,crossover_population,generate_population

from sklearn.metrics import f1_score,accuracy_score,roc_auc_score, mean_squared_error, r2_score

import numpy as np
import random
from tqdm import tqdm

import pandas as pd

import ydf

random.seed(123)
np.random.seed(123)

def data_chromosome_subset(X, chromosome):
    return X.loc[:, chromosome.ravel() == 1]

def fitness_score(X_tr, y_tr, X_te, y_te, chromosome, metric):
    np.random.seed(123); random.seed(123)
    # reshape chromosome incase it has wrong dimension...
    X_tr_subset = data_chromosome_subset(X_tr, chromosome)
    X_te_subset = data_chromosome_subset(X_te, chromosome)

    target_series_tr = y_tr['^GSPC']
    target_series_te = y_te['^GSPC']

    train_df = pd.concat([X_tr_subset, target_series_tr.rename('sp500_target')], axis=1)
    # set features IMPT
    features_for_training = X_tr_subset.columns.tolist()
    train_df_for_ydf = train_df[features_for_training + ['sp500_target']]
    
    model = ydf.RandomForestLearner(label='sp500_target', task=ydf.Task.REGRESSION).train(train_df_for_ydf)
    y_pr = model.predict(X_te_subset)

    return mean_directional_accuracy(target_series_te, y_pr)

def fitness_population(X_tr,y_tr,X_te,y_te,population,metric,verbose=False):
    n_chromosomes,n_genes = population.shape
    scores = np.empty((n_chromosomes,),dtype=float)

    if verbose:
        slice = tqdm(range(n_chromosomes))
    else:
        slice = range(n_chromosomes)

    for n in slice:
        scores[n] = fitness_score(X_tr,y_tr,X_te,y_te,population[[n],:],metric)

    return scores

def chromosome_selection(scores, population, epsilon):
    n_chromosomes, n_genes = population.shape
    n_select = n_chromosomes // 2
    selected_chromosomes = []

    # Indices of chromosomes sorted by score in descending order (for exploitation)
    sorted_indices = np.argsort(scores)[::-1]

    # Number of chromosomes to select randomly (exploration)
    n_explore = int(epsilon * n_select)

    # Number of chromosomes to select based on fitness (exploitation)
    n_exploit = n_select - n_explore

    # Select random chromosomes for exploration
    random_indices = random.sample(range(n_chromosomes), n_explore)
    selected_chromosomes.extend(population[random_indices])

    # Select top-performing chromosomes for exploitation
    top_indices = sorted_indices[:n_exploit]
    selected_chromosomes.extend(population[top_indices])

    return np.array(selected_chromosomes)


def generate_next_population(scores,population,crossover_method="onepoint",mutation_rate=0.1,elitism=2, epsilon=0.1):
    n_chromosomes,n_genes = population.shape
    chromosome_fittest = chromosome_selection(scores, population, epsilon)

    # cross over RANDOMLY!!!
    chromosome_cross = crossover_population(chromosome_fittest,n_chromosomes//2-elitism,crossover_method)
    # mutation
    chromosome_mutate = mutation(chromosome_fittest,mutation_rate)

    return np.vstack((chromosome_fittest[:elitism,:],chromosome_cross,chromosome_mutate))

def mean_directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2 or len(y_pred) < 2 or len(y_true) != len(y_pred):
        return 0.0

    correct_direction_predictions = 0
    total_directional_predictions = len(y_true) - 1

    for i in range(1, len(y_true)):
        actual_change = y_true[i] - y_true[i-1]
        predicted_change = y_pred[i] - y_pred[i-1]

        if (actual_change > 0 and predicted_change > 0) or \
           (actual_change < 0 and predicted_change < 0) or \
           (actual_change == 0 and predicted_change == 0): #unlikely
            correct_direction_predictions += 1

    if total_directional_predictions == 0:
        return 0.0
    else:
        return correct_direction_predictions / total_directional_predictions

def select_metric(metric_choice):
    if metric_choice == "accuracy":
        return accuracy_score
    elif metric_choice == "f1":
        return f1_score
    elif metric_choice == "roc_auc_score":
        return roc_auc_score
    elif metric_choice == "MSE":
        return mean_squared_error
    elif metric_choice == "R2":
        return r2_score

def genetic_algorithm_feature_selection(X, y, metric, n_generations=200, n_chromosomes=50,
                                        crossover_method="multipoint", mutation_rate=0.15, elitism=2,
                                        verbose=True, early_stopping_rounds=None):
    np.random.seed(123);
    random.seed(123)
    # Get the number of features
    n_features = X.shape[1]

    # Initialize the population
    population = generate_population(n_chromosomes, n_features)

    # Select the metric function
    metric_function = select_metric(metric)

    # Split the data into training and testing sets
    split_index = int(0.8 * len(X))
    
    X_train = X.iloc[:split_index]
    y_train = y[:split_index] 
    X_test = X.iloc[split_index:]
    y_test = y[split_index:]  


    best_scores = []
    best_chromosome = None
    best_score = float('-inf')

    # Early stopping variables
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        consecutive_no_improvement = 0
        previous_best_score = float('-inf')
    else:
        early_stopping_rounds = None  # Ensure it's None if not positive

    # Main genetic algorithm loop
    for generation in range(n_generations):
        # Evaluate the fitness of the current population
        scores = fitness_population(X_train, y_train, X_test, y_test, population, metric_function, verbose=verbose)

        current_best_index = np.argmax(scores)
        current_best_score = scores[current_best_index]

        if best_chromosome is None or current_best_score > best_score:
            best_score = current_best_score
            best_chromosome = population[current_best_index].copy()
            if early_stopping_rounds is not None:
                consecutive_no_improvement = 0  # Reset counter on improvement
        elif early_stopping_rounds is not None and current_best_score == best_score:
            consecutive_no_improvement += 1

        best_scores.append(current_best_score)

        if verbose:
            print(f"Generation {generation + 1}/{n_generations} - Best Score: {best_score:.4f}")

        # Check for early stopping
        if early_stopping_rounds is not None and consecutive_no_improvement >= early_stopping_rounds:
            if verbose:
                print(f"\nEarly stopping triggered after {generation + 1} generations "
                      f"with no improvement for {early_stopping_rounds} rounds.")
            break

        # Generate the next population
        population = generate_next_population(scores, population, crossover_method, mutation_rate, elitism)

    if verbose:
        print("\nGenetic Algorithm Finished!")
        print(f"Best Chromosome (Feature Selection): {best_chromosome}")
        print(f"Best Score: {best_score:.4f}")

    return best_chromosome, best_score, best_scores