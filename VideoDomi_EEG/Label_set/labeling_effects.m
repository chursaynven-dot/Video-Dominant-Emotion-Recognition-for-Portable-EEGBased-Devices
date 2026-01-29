clc; clear; close all;
%% This script provides an unsupervised clustering-based exploratory analysis 
 % to assess emotion consistency and dominant patterns at the video and subject levels
%% Load data
load("for_effects.mat");          % variables: emo
num_subjects = 32;
num_videos   = 40;

%% Random seed
try
    rng(42,'twister');
catch
    rand('twister',42);
end

%% Feature preparation
data = emo(:,1:8);                % [Subject, Video, V, A, D, L, ...]
feat_end = 6;
features = zscore(data(:,3:feat_end));

%% PCA (retain 95% variance)
[coeff, score, ~, ~, explained] = pca(features);
dim95 = find(cumsum(explained) >= 95, 1);
reduced_data = [score(:,1:dim95), features];

%% Clustering comparison
methods = {'kmeans','gmm','hierarchical'};
labels_all = cell(1,length(methods));
sil_scores = zeros(1,length(methods));
conf_scores = zeros(1,length(methods));

for i = 1:length(methods)
    switch methods{i}
        case 'kmeans'
            labels = kmeans(reduced_data,4,'Replicates',8);

        case 'gmm'
            gm = fitgmdist(reduced_data,4, ...
                'Replicates',15, ...
                'RegularizationValue',1e-4, ...
                'CovarianceType','full');
            labels = cluster(gm,reduced_data);

        case 'hierarchical'
            Z = linkage(reduced_data,'ward');
            labels = cluster(Z,'maxclust',4);
    end

    labels_all{i} = labels;

    % Video-level dominant labels
    video_dom = zeros(1,num_videos);
    conformity = zeros(num_subjects,1);

    for v = 1:num_videos
        v_labels = labels(data(:,2) == v);
        if ~isempty(v_labels)
            video_dom(v) = mode(v_labels);
        end
    end

    for s = 1:num_subjects
        idx = data(:,1) == s;
        conformity(s) = mean(labels(idx) == video_dom(data(idx,2))');
    end

    conf_scores(i) = mean(conformity);
    sil_scores(i)  = mean(silhouette(reduced_data,labels));
end

%% Method selection
sil_norm  = rescale(sil_scores);
conf_norm = rescale(conf_scores);
final_score = 0.3*sil_norm + 0.7*conf_norm;

[~, best_idx] = max(final_score);
labels = labels_all{best_idx};
best_method = methods{best_idx};

fprintf('\n=== Clustering Comparison ===\n');
for i = 1:length(methods)
    fprintf('%-12s : Silhouette = %.3f, Conformity = %.2f%%\n', ...
        methods{i}, sil_scores(i), conf_scores(i)*100);
end
fprintf('Selected method: %s\n\n', upper(best_method));

%% Video-level statistics
video_dom = zeros(1,num_videos);
video_cons = zeros(1,num_videos);
video_dist = zeros(num_videos,4);

for v = 1:num_videos
    v_idx = data(:,2) == v;
    v_labels = labels(v_idx);
    if isempty(v_labels), continue; end

    for e = 1:4
        video_dist(v,e) = sum(v_labels == e) / numel(v_labels);
    end

    [m_val, m_cnt] = mode(v_labels);
    video_dom(v)  = m_val;
    video_cons(v) = m_cnt / numel(v_labels);
end

%% Individual conformity
conformity = zeros(num_subjects,1);
for s = 1:num_subjects
    idx = data(:,1) == s;
    conformity(s) = mean(labels(idx) == video_dom(data(idx,2))');
end

mean_conformity = mean(conformity);
fprintf('Mean conformity: %.1f%%\n', mean_conformity*100);

%% Outlier detection
threshold = 0.5;
outliers = find(conformity < threshold);
fprintf('Outliers (conformity < %.0f%%): %s\n', ...
    threshold*100, mat2str(outliers'));

%% Visualization settings
set(0,'DefaultFigureColor','w', ...
      'DefaultAxesFontName','Arial', ...
      'DefaultAxesFontSize',10);

%% Figure 1: Individual conformity
figure('Position',[100 100 600 400]);
[~, idx] = sort(conformity,'descend');
bar(conformity(idx)*100);
hold on;
plot(xlim,[threshold threshold]*100,'r--','LineWidth',1.2);
xlabel('Participant (sorted)');
ylabel('Conformity (%)');
title(sprintf('Individual Conformity (Mean = %.1f%%)', mean_conformity*100));
ylim([0 100]); grid on; box off;

%% Figure 2: PCA visualization
figure('Position',[100 100 700 500]);
scatter(score(:,1),score(:,2),15,labels,'filled');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title(['PCA Visualization (' upper(best_method) ')']);
grid on; box off;
