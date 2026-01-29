clc; clear; close all;

%% Data loading
data_file = 'example.xlsx';
[num_data, ~, ~] = xlsread(data_file);

X = num_data(:, 1:151);                % Feature matrix
y = categorical(num_data(:, 152));     % Class labels
subject_ids = num_data(:, 153);        % Subject identifiers

%% Subject-wise normalization
[X_norm, ~] = normalizeBySubject(X, subject_ids);

%% Feature augmentation (temporal difference)
X_diff = diff(X_norm, 1, 2);
X_aug = [X_norm(:, 1:end-1), X_diff];

%% Feature selection (MRMR)
feat_idx = fscmrmr(X_aug, y);
X_sel = X_aug(:, feat_idx(1:80));

%% Subject-independent cross-validation
subjects = unique(subject_ids);
cv = cvpartition(numel(subjects), 'KFold', 5);

acc_svm = zeros(1, cv.NumTestSets);
acc_rf  = zeros(1, cv.NumTestSets);
acc_dnn = zeros(1, cv.NumTestSets);

for k = 1:cv.NumTestSets
    
    test_subj  = subjects(cv.test(k));
    train_mask = ~ismember(subject_ids, test_subj);
    test_mask  = ismember(subject_ids, test_subj);
    
    X_train = X_sel(train_mask, :);
    y_train = y(train_mask);
    X_test  = X_sel(test_mask, :);
    y_test  = y(test_mask);
    
    %% SVM classifier
    svm_model = fitcecoc( ...
        X_train, y_train, ...
        'Learners', templateSVM('KernelFunction','rbf', ...
                                'KernelScale','auto', ...
                                'Standardize', false));
    acc_svm(k) = mean(predict(svm_model, X_test) == y_test);
    
    %% Random Forest classifier
    rf_model = fitcensemble( ...
        X_train, y_train, ...
        'Method','Bag', ...
        'NumLearningCycles',200, ...
        'Learners', templateTree('MaxNumSplits',20));
    acc_rf(k) = mean(predict(rf_model, X_test) == y_test);
    
    %% Deep Neural Network classifier
    layers = [
        featureInputLayer(size(X_train,2))
        fullyConnectedLayer(256)
        batchNormalizationLayer
        reluLayer
        dropoutLayer(0.5)
        
        fullyConnectedLayer(128)
        batchNormalizationLayer
        reluLayer
        
        fullyConnectedLayer(numel(categories(y)))
        softmaxLayer
        classificationLayer ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',50, ...
        'MiniBatchSize',64, ...
        'InitialLearnRate',1e-4, ...
        'L2Regularization',1e-3, ...
        'Shuffle','every-epoch', ...
        'Verbose',false);
    
    dnn_model = trainNetwork(X_train, y_train, layers, options);
    acc_dnn(k) = mean(classify(dnn_model, X_test) == y_test);
    
    fprintf('Fold %d | SVM: %.2f%% | RF: %.2f%% | DNN: %.2f%%\n', ...
        k, acc_svm(k)*100, acc_rf(k)*100, acc_dnn(k)*100);
end

%% Subject-wise normalization function
function [X_norm, stats] = normalizeBySubject(X, subject_ids)
    subjects = unique(subject_ids);
    X_norm = zeros(size(X));
    stats = struct();
    
    for i = 1:numel(subjects)
        idx = subject_ids == subjects(i);
        [X_norm(idx,:), mu, sigma] = zscore(X(idx,:));
        stats(i).id = subjects(i);
        stats(i).mean = mu;
        stats(i).std  = sigma;
    end
end
