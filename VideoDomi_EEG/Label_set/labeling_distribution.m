clc; clear; close all;
%% The released code implements the video-level emotion reorganization and consistency-based labeling strategy
%% Configuration
mode = 'filter';                 % 'major' | 'filter' | 'soft'
consistency_threshold = 0.5;

num_subjects = 32;
num_videos   = 40;

%% Load DEAP rating data
load('for_distribution.mat');    % variable: emo (N x 4) [V A D L]
data = emo;

V = data(:,1);
A = data(:,2);
D = data(:,3);
L = data(:,4);

num_samples = num_subjects * num_videos;

%% Subject-wise z-score normalization
z_data = zeros(size(data));
for s = 1:num_subjects
    idx = (s-1)*num_videos + (1:num_videos);
    z_data(idx,:) = zscore(data(idx,:));
end

%% Emotion labeling (VA-based with D/L adjustment)
sample_labels = zeros(num_samples,1);

for i = 1:num_samples
    zv = z_data(i,1);
    za = z_data(i,2);
    zd = z_data(i,3);
    zl = z_data(i,4);

    if zv >= 0 && za >= 0
        base_label = 1;   % high V, high A
    elseif zv >= 0 && za < 0
        base_label = 2;   % high V, low A
    elseif zv < 0 && za < 0
        base_label = 3;   % low V, low A
    else
        base_label = 4;   % low V, high A
    end

    if (base_label <= 2) && (zd < -1 || zl < -1)
        sample_labels(i) = base_label + 0.1;
    elseif (base_label >= 3) && (zd > 1 || zl > 1)
        sample_labels(i) = base_label - 0.1;
    else
        sample_labels(i) = base_label;
    end
end

sample_labels = round(sample_labels);

%% Video-level emotion statistics
video_labels = reshape(sample_labels, num_videos, num_subjects)';
video_emotion_counts = zeros(num_videos,4);

for v = 1:num_videos
    for e = 1:4
        video_emotion_counts(v,e) = sum(video_labels(:,v) == e);
    end
end

%% Dominant emotion and consistency
[video_major_count, video_major_emotion] = max(video_emotion_counts,[],2);
video_consistency = video_major_count / num_subjects;

%% Dominant emotion voters
major_voters = cell(num_videos,1);
for v = 1:num_videos
    major_voters{v} = find(video_labels(:,v) == video_major_emotion(v));
end

%% Label assignment modes
switch mode
    case 'major'
        sample_labels = reshape( ...
            video_major_emotion(repmat(1:num_videos,num_subjects,1)), [], 1);

    case 'filter'
        low_consistency = video_consistency < consistency_threshold;
        sample_labels = reshape(video_labels,[],1);
        sample_labels( ...
            ismember(ceil((1:num_samples)'/num_subjects), find(low_consistency))) = 0;

    case 'soft'
        sample_labels = zeros(num_samples,4);
        for v = 1:num_videos
            prob = video_emotion_counts(v,:) / sum(video_emotion_counts(v,:));
            sample_labels(v:num_videos:end,:) = repmat(prob,num_subjects,1);
        end
end

%% Summary
fprintf('Average consistency: %.2f%%\n', mean(video_consistency)*100);
fprintf('Videos below threshold (%.2f): %d / %d\n', ...
    consistency_threshold, sum(video_consistency < consistency_threshold), num_videos);

emo_dist = histcounts(sample_labels(sample_labels>0),1:5);
emo_dist = emo_dist / sum(emo_dist);
fprintf('Emotion ratio: %.2f %.2f %.2f %.2f\n', emo_dist);
