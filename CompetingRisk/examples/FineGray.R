# This script runs Fine Gray on all datasets

library(riskRegression)
library(prodlim)
library(survival)
library(cmprsk)
library(readr)

# Open data
#datasets = c('SEER', 'FRAMINGHAM', 'SYNTHETIC_COMPETING', 'PBC')
datasets = c('ABDOMEN_EMBEDDINGS_PCA')
n_outcomes = 4

for (dataset in datasets) {
    # Open saved train file
    data = read_csv(paste0('./data/', dataset, '.csv'))
    #col_to_remove <-c('featurefeature_590')
    #data <- data[, !(names(data) %in% col_to_remove)]

    # Create matrix results for both Fine Gray and Cox
    prediction_fg = matrix(0, nrow = nrow(data), ncol = n_outcomes * 100 + 1) # outcomes * Times
    rownames(prediction_fg) = as.numeric(rownames(data)) - 1
    eval_times = seq(min(data['Time']), max(data['Time']), length.out = 100)
    colnames(prediction_fg) = c(rep(eval_times, n_outcomes), 'Use')
    prediction_cs = prediction_fg # Make a copy

    var_tot = setdiff(colnames(data), c("Time", "Event", "Fold_0", "Fold_1", "Fold_2", "Fold_3", "Fold_4"))

    # Cross validation
    for (fold in 0:4) {
        print(fold)
        data_folder = subset(data, data[paste0("Fold_", fold)] == "Train")[append(var_tot, c('Time', 'Event'))]

        # Create associated formula
        var = setdiff(colnames(data_folder), c("Time", "Event"))
        formula = reformulate(var, response = "Hist(Time, Event)") 
        if (dataset == 'FRAMINGHAM') {
            formula = reformulate(setdiff(colnames(data_folder), c("Time", "Event", 'feature8')), response = "Hist(Time, Event)") 
            if (fold == 0){
                formula = reformulate(setdiff(colnames(data_folder), c("Time", "Event", 'feature6', 'feature7', 'feature8')), response = "Hist(Time, Event)") 
            }
        }
        print(formula)

        # Fit model 
        for (outcome in 1:n_outcomes) {
            test = (data[paste0("Fold_", fold)] == "Test")

            # Run Fine Gray
            tryCatch(
                expr = {
                        model = FGR(formula, data = data_folder, cause = outcome)
                        # Predict at the time horizons of interest CSC + predictRisk
                        prediction_fg[test,((outcome-1)*100+1):(outcome*100)] = 1 - predict(model, subset(data, test), eval_times, cause = outcome)
                    },
                error = function(e){ 
                    print(e)
                    prediction_fg[test,((outcome-1)*100+1):(outcome*100)] = NA
                })
            prediction_fg[(data[paste0("Fold_", fold)] == "Test"), ncol(prediction_fg)] = fold

            # Run Cause specific Cox
            tryCatch(
                expr = {
                        model = CSC(formula, data = data_folder, cause = outcome, method = "breslow", fitter = "cph", iter = 100)
                        # Predict at the time horizons of interest CSC + predictRisk
                        prediction_cs[test,((outcome-1)*100+1):(outcome*100)] = 1 - predictRisk(model, subset(data, test), eval_times, cause = outcome)
                    },
                error = function(e){ 
                    print(e)
                    prediction_cs[test,((outcome-1)*100+1):(outcome*100)] = NA
                })
            prediction_cs[(data[paste0("Fold_", fold)] == "Test"), ncol(prediction_cs)] = fold
        }
    }
    # Save
    write.csv(prediction_fg, paste0('Results_04.06/', dataset, '_finegray.csv'))
    write.csv(prediction_cs, paste0('Results_04.06/', dataset, '_coxcs.csv'))
}