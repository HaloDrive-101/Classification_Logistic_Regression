clear all; close all; clc;

data = load("QSAR_data.mat");
variable_names = fieldnames(data);
matrix_data = data.(variable_names{1});
reg = [];                                                          % array for learning curve

% Separate features and target
X = matrix_data(:, 1:41);                                          % Features
Y = matrix_data(:, 42);                                            % Target (0 or 1)

% Z-score Normalization
mean_data = mean(X);
std_data = std(X, 1);
norm_feature = (X - mean_data) ./ std_data;
norm_feature = [ones(size(norm_feature, 1), 1), norm_feature];     % Add bias term

% K-Fold Cross-Validation Setup
k = 5;                                                             % Number of folds
cv = cvpartition(Y, 'KFold', k, 'Stratify', true);                 % Stratified k-fold partitioning

% Initialize variables for data and performace parameters
theta_kfold = zeros(size(norm_feature, 2), k);                     
auc_fold = zeros(k, 1);                                           
sensitivity = zeros(k, 1);
specificity = zeros(k, 1);
accuracy = zeros(k, 1);
precision = zeros(k, 1);
recall = zeros(k, 1);
f1_score = zeros(k, 1);

% Training Hyperparameters
lambda = 0.01;                                                     % Regularization parameter
max_itr = 20;                                                      % Maximum iterations
tol = 1e-6;                                                        % Convergence tolerance

% Performing K-Fold Cross-Validation
figure;                                                            

for fold = 1:k
    % Fetching training and validation indices for ongoing fold
    train_idx = training(cv, fold);
    val_idx = test(cv, fold);
   
    % Splitting the data into training and validation sets for ongoing fold
    X_train_set = norm_feature(train_idx, :);
    Y_train_set = Y(train_idx);
    X_val_set   = norm_feature(val_idx, :);
    Y_val_set   = Y(val_idx);
   
    % Initializing weights
    [m, n] = size(X_train_set);                                    % m not used 
    theta = zeros(n, 1);                                           % Creating 1D matrix for theta where future updates will happen
   
    % Newton's Method with L1 Regularization
    for itr = 1:max_itr
        z = X_train_set * theta;
        f = 1 ./ (1 + exp(-z));                                    % Sigmoid function
       
        % Computing gradient with L1 regularization
        grad = X_train_set' * (f - Y_train_set);
        l1_gradient = lambda * sign(theta);                        % Multiplying by direction of coefficient
        l1_gradient(1) = 0;                                        % Excluding bias term from regularization ( 'b' in z  = (wx+b) not required to regularize)
        reg_grad = grad + l1_gradient;                             % Adding regularization term
        
        % Computing the Hessian matrix
        W = diag(f .* (1 - f));
        H = X_train_set' * W * X_train_set;
       
        % Updating theta
        theta = theta - H \ reg_grad;

        % Checking  convergence
        if norm(reg_grad, 2) < tol
            break;
        end
    end
   
    % Storing the weights for this fold
    theta_kfold(:, fold) = theta;
   
    % To evaluate the validation set
    z_val = X_val_set * theta;
    y_pred_prob = 1 ./ (1 + exp(-z_val));                          % Predicted probabilities
    y_pred = round(y_pred_prob);                                   % Converting probabilities to 0/1 predictions
   
    % Computing confusion matrix
    tp = sum((Y_val_set == 1) & (y_pred == 1));                    % True Positive
    tn = sum((Y_val_set == 0) & (y_pred == 0));                    % True Negative
    fp = sum((Y_val_set == 0) & (y_pred == 1));                    % False Positive
    fn = sum((Y_val_set == 1) & (y_pred == 0));                    % False Negative
   
    % Computing foldwise performace parameters 
    sensitivity(fold) = tp / max(tp + fn, 1);
    specificity(fold) = tn / max(tn + fp, 1);
    accuracy(fold) = (tp + tn) / length(Y_val_set);
    precision(fold) = tp / max(tp + fp, 1);
    recall(fold) = tp / max(tp + fn, 1);
    f1_score(fold) = 2 * (precision(fold) * recall(fold)) / max(precision(fold) + recall(fold), 1);

    % Computing ROC curve and AUC for the validation set
    subplot(2, 3, fold);                                         
    [roc_x, roc_y, ~, auc_fold(fold)] = perfcurve(Y_val_set, y_pred_prob, 1);
    plot(roc_x, roc_y, '-b', 'LineWidth', 2);
    title(['ROC Curve Fold ', num2str(fold), ', AUC = ', num2str(auc_fold(fold))]);
    xlabel('False Positive Rate (FPR)');
    ylabel('True Positive Rate (TPR)');
    grid on;
end

% Finding mean of the K-Fold Results
mean_auc = mean(auc_fold);
mean_sensitivity = mean(sensitivity);
mean_specificity = mean(specificity);
mean_accuracy = mean(accuracy);
mean_precision = mean(precision);
mean_recall = mean(recall);
mean_f1_score = mean(f1_score);

disp('K-Fold Cross-Validation Results:');
disp(['Mean Sensitivity: ', num2str(mean_sensitivity)]);
disp(['Mean Specificity: ', num2str(mean_specificity)]);
disp(['Mean Accuracy: ', num2str(mean_accuracy)]);
disp(['Mean Precision: ', num2str(mean_precision)]);
disp(['Mean Recall: ', num2str(mean_recall)]);
disp(['Mean F1 Score: ', num2str(mean_f1_score)]);
disp(['Mean AUC: ', num2str(mean_auc)]);

% Training Final Model on Entire Dataset
[m, n] = size(norm_feature);
theta = zeros(n, 1);
for itr = 1:max_itr
    z = norm_feature * theta;
    f = 1 ./ (1 + exp(-z));
    grad = norm_feature' * (f - Y);
    l1_gradient = lambda * sign(theta);
    l1_gradient(1) = 0; % Exclude bias term
    reg_grad = grad + l1_gradient;
    reg = [reg,reg_grad];
    W = diag(f .* (1 - f));
    H = norm_feature' * W * norm_feature;
    theta = theta - H \ reg_grad;
    if norm(grad, 2) < tol
        break;
    end
end

disp('Final Model Trained on Entire Dataset');

% Evaluating Final Model on Entire Dataset
z_final = norm_feature * theta;
y_final_pred_prob = 1 ./ (1 + exp(-z_final));
y_final_pred = round(y_final_pred_prob);

% Final Confusion Matrix
tp_final = sum((Y == 1) & (y_final_pred == 1));                  % True Positive
tn_final = sum((Y == 0) & (y_final_pred == 0));                  % True Negative
fp_final = sum((Y == 0) & (y_final_pred == 1));                  % False Positive
fn_final = sum((Y == 1) & (y_final_pred == 0));                  % False Negative

% Final Performance Metrics
sensitivity_final = tp_final / (tp_final + fn_final);
specificity_final = tn_final / (tn_final + fp_final);
accuracy_final = (tp_final + tn_final) / length(Y);
precision_final = tp_final / (tp_final + fp_final);
recall_final = tp_final / (tp_final + fn_final);
f1_final = 2 * (precision_final * recall_final) / (precision_final + recall_final);

disp('Final Model Results:');
disp(['Sensitivity: ', num2str(sensitivity_final)]);
disp(['Specificity: ', num2str(specificity_final)]);
disp(['Accuracy: ', num2str(accuracy_final)]);
disp(['Precision: ', num2str(precision_final)]);
disp(['Recall: ', num2str(recall_final)]);
disp(['F1 Score: ', num2str(f1_final)]);


% Computing the Final ROC Curve and AUC
subplot(2, 3, 6);  
[roc_x_final, roc_y_final, ~, auc_final] = perfcurve(Y, y_final_pred_prob, 1);
plot(roc_x_final, roc_y_final, '-r', 'LineWidth', 2);
title(['Final ROC Curve (Entire Dataset), AUC = ', num2str(auc_final)]);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
grid on;

disp(['Final AUC (Entire Dataset): ', num2str(auc_final)]);

% Confusion Matrix for the final model
figure;
confusionchart(Y, y_final_pred); % True labels vs predicted labels
title('Confusion Matrix for Final Model');
 
% Learning Curve
figure;
plot(1:max_itr, reg, '-b', 'LineWidth', 2);  
xlabel('Iteration');
ylabel('Loss Function');
legend('Features Loss Function');
title('Learning Curve: loss Functions of Features');
grid on;

% Comparison of mean values from k-fold and final values
mean_values = [mean_sensitivity, mean_specificity, mean_accuracy, mean_precision, mean_f1_score];
final_values = [sensitivity_final, specificity_final, accuracy_final, precision_final, f1_final];
figure;
bar([mean_values', final_values'], 'grouped');
set(gca, 'XTickLabel', {'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'F1 Score', 'AUC'}, 'XTick', 1:6);
ylabel('Values');
legend('Mean', 'Final', 'Location', 'Best');
title('Comparison of Mean and Final Metrics');
grid on;

% Before and After Normalization plot
figure;
% First subplot
subplot(1, 2, 1); 
scatter(X(:,12),X(:,30), 'filled');
axis([-5 5 -2 80]);
title('Before Normalization');
xlabel('Feature 8');
ylabel('Feature 28');
grid on;
% Second subplot
subplot(1, 2, 2); 
scatter(norm_feature(:,13), norm_feature(:,31), 'filled'); % Scatter plot of normalized data
axis([-5 5 -1 5.5]);
title('After Z-Score Normalization');
xlabel('Feature 8 (Normalized)');
ylabel('Feature 28 (Normalized)');
grid on;
