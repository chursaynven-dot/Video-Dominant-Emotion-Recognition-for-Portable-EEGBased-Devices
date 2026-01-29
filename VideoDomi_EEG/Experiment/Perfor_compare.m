clc; clear; close all;

%% 1. Load data and preprocessing (assume column 152 is label, column 153 is subject ID)
data_file = 'example.xlsx';
[num_data, ~, ~] = xlsread(data_file);

X = num_data(:, 1:151);            % Feature data
y = categorical(num_data(:, 152)); % Labels converted to categorical
subject_ids = num_data(:, 153);    % Subject ID column

%% 2. Feature engineering
% 2.1 Normalize by subject
[X_normalized, subject_mean] = normalizeBySubject(X, subject_ids);

% 2.2 Differential features
X_diff = diff(X_normalized, 1, 2); 
X_augmented = [X_normalized(:, 1:end-1), X_diff];

% 2.3 Feature selection
feat_selector = fscmrmr(X_augmented, y);
selected_feats = feat_selector(1:80); % Select top 80 important features
X_final = X_augmented(:, selected_feats);

%% 3. Subject-independent cross-validation (5-fold)
unique_subjects = unique(subject_ids);
cv = cvpartition(length(unique_subjects), 'KFold', 5);

accuracy_svm = zeros(1, cv.NumTestSets);
accuracy_rf = zeros(1, cv.NumTestSets);
accuracy_dnn = zeros(1, cv.NumTestSets);

% Store all predictions and ground truth for confusion matrix
all_true = [];
all_pred_svm = [];
all_pred_rf = [];
all_pred_dnn = [];

for fold = 1:cv.NumTestSets
    % Split train/test by subject
    test_subjects = unique_subjects(cv.test(fold));
    train_mask = ~ismember(subject_ids, test_subjects);
    test_mask = ismember(subject_ids, test_subjects);
    
    X_train = X_final(train_mask, :);
    y_train = y(train_mask);
    X_test = X_final(test_mask, :);
    y_test = y(test_mask);
    
    %% Model 1: SVM
    svm_template = templateSVM('KernelFunction', 'rbf', ...
                              'Standardize', false, ...
                              'KernelScale', 'auto', ...
                              'BoxConstraint', 1);
    model_svm = fitcecoc(X_train, y_train, 'Learners', svm_template);
    y_pred_svm = predict(model_svm, X_test);
    accuracy_svm(fold) = sum(y_pred_svm == y_test) / numel(y_test);
    
    %% Model 2: Random Forest
    model_rf = fitcensemble(X_train, y_train, ...
                          'Method', 'Bag', ...
                          'NumLearningCycles', 200, ...
                          'Learners', templateTree('MaxNumSplits', 20));
    y_pred_rf = predict(model_rf, X_test);
    accuracy_rf(fold) = sum(y_pred_rf == y_test) / numel(y_test);
    
    %% Model 3: DNN
    layers = [
        featureInputLayer(size(X_train, 2))
        fullyConnectedLayer(256, 'Name', 'fc1')
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5)
        
        fullyConnectedLayer(128, 'Name', 'fc2')
        batchNormalizationLayer
        reluLayer
        
        fullyConnectedLayer(numel(categories(y)), 'Name', 'output')
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.0001, ...
        'L2Regularization', 0.001, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false);
    
    model_dnn = trainNetwork(X_train, y_train, layers, options);
    y_pred_dnn = classify(model_dnn, X_test);
    accuracy_dnn(fold) = sum(y_pred_dnn == y_test) / numel(y_test);
    
    % Save predictions
    all_true = [all_true; y_test];
    all_pred_svm = [all_pred_svm; y_pred_svm];
    all_pred_rf = [all_pred_rf; y_pred_rf];
    all_pred_dnn = [all_pred_dnn; y_pred_dnn];
    
end

%% 4. Classification metrics (Precision, Recall, F1)
metrics_svm = classification_metrics(all_true, all_pred_svm);
metrics_rf  = classification_metrics(all_true, all_pred_rf);
metrics_dnn = classification_metrics(all_true, all_pred_dnn);

disp('=== SVM Metrics ==='); disp(metrics_svm);
disp('=== RF Metrics ===');  disp(metrics_rf);
disp('=== DNN Metrics ==='); disp(metrics_dnn);

%% Helper function: Normalize features by subject
function [X_norm, subject_mean] = normalizeBySubject(X, subject_ids)
    unique_subjects = unique(subject_ids);
    X_norm = zeros(size(X));
    subject_mean = struct();
    
    for i = 1:length(unique_subjects)
        subj_mask = (subject_ids == unique_subjects(i));
        [X_norm(subj_mask, :), mu, sigma] = zscore(X(subj_mask, :));
        subject_mean(i).id = unique_subjects(i);
        subject_mean(i).mu = mu;
        subject_mean(i).sigma = sigma;
    end
end

%% Helper function: Compute Precision / Recall / F1
function metrics = classification_metrics(y_true, y_pred)
    classes = categories(y_true);
    n_class = numel(classes);
    metrics = table('Size',[n_class 6], ...
                    'VariableTypes',{'string','double','double','double','double','double'}, ...
                    'VariableNames',{'Class','Precision','Recall','F1','TP','TN'});
    N = numel(y_true);
    for i = 1:n_class
        c = classes{i};
        TP = sum(y_true==c & y_pred==c);
        FP = sum(y_true~=c & y_pred==c);
        FN = sum(y_true==c & y_pred~=c);
        TN = N - TP - FP - FN;
        Precision = TP / (TP+FP+eps);
        Recall = TP / (TP+FN+eps);
        F1 = 2*Precision*Recall/(Precision+Recall+eps);
        metrics(i,:) = {c, Precision, Recall, F1, TP, TN};
    end
end
