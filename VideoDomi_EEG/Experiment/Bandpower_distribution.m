clc; clear; close all;

%% Parameter Settings
fs = 128;
num_channels = 11;
useGPU = false;

bands = {
    'delta', 1, 4;
    'theta', 4, 8;
    'alpha', 8, 14;
    'beta', 14, 30;
    'gamma', 30, 60;
};
n_bands = size(bands, 1);
channel_names = {'FP1','FP2','F3','F4','C3','C4','T7','T8','Pz','O1','O2'};

%% Set file path
data_folder = '\Video_e_g\Emo1\';
file_list = dir(fullfile(data_folder, '*.mat'));
num_files = length(file_list);

all_eeg_data = cell(num_files, 1);
min_length = inf;

fprintf('=== Start loading data ===\n');

for i = 1:num_files
    mat = load(fullfile(file_list(i).folder, file_list(i).name));
    vars = fieldnames(mat);
    eeg_found = false;

    for k = 1:length(vars)
        temp = mat.(vars{k});
        if isnumeric(temp) && size(temp,1)==11
            eeg = double(temp);
            eeg_found = true;
            break;
        end
    end

    if ~eeg_found
        warning('Skipping %s, 11-channel matrix not found.', file_list(i).name);
        continue;
    end

    all_eeg_data{i} = eeg;
    min_length = min(min_length, size(eeg,2));
    fprintf('Loaded %s | Size: [%d × %d]\n', file_list(i).name, size(eeg,1), size(eeg,2));
end

valid_data = ~cellfun(@isempty, all_eeg_data);
eeg_tensor = zeros(11, min_length, sum(valid_data));

idx = 1;
for i = 1:num_files
    if isempty(all_eeg_data{i}), continue; end
    eeg_tensor(:,:,idx) = all_eeg_data{i}(:,1:min_length);
    idx = idx + 1;
end
eeg_median = median(eeg_tensor, 3);   % [11 × min_length]
fprintf('=== Successfully loaded %d valid samples ===\n', idx-1);

%% Load FastICA
addpath('G:\F项目资料夹\研二下\20250322脑电EEG\整理\FastICA_2.5\FastICA_25');

% Initialize results
stft_energy_total = zeros(num_channels, n_bands);
wavelet_energy_total = zeros(num_channels, n_bands);

for sample_idx = 1:size(eeg_tensor, 3)
    eeg = eeg_median;

    % Band-pass filter
    [b_bp, a_bp] = butter(4, [1 45]/(fs/2), 'bandpass');
    eeg_bp = filtfilt(b_bp, a_bp, eeg_median')';

    % Notch filter (50Hz)
    d = designfilt('bandstopiir','FilterOrder',4, ...
        'HalfPowerFrequency1',49,'HalfPowerFrequency2',51, ...
        'DesignMethod','butter','SampleRate',fs);
    eeg_notch = filtfilt(d, eeg_bp')';

    % Common average reference (CAR)
    eeg_car = eeg_notch - mean(eeg_notch,1);

    % FastICA
    [icasig, A, W] = fastica(eeg_car, 'approach', 'symm', 'g', 'tanh', ...
        'lastEig', num_channels, 'numOfIC', num_channels, ...
        'stabilization', 'on', 'verbose', 'off');

    % Artifact removal using kurtosis
    kurt_values = kurtosis(icasig, 1, 2);
    artifact_components = find(abs(kurt_values) > 5);
    icasig(artifact_components,:) = 0;
    eeg_ica = A * icasig;

    if useGPU, eeg_ica = gpuArray(eeg_ica); end

    %% Frequency band energy analysis (median)
    stft_energy_median = zeros(num_channels, n_bands);
    wavelet_energy_median = zeros(num_channels, n_bands);

    % STFT analysis
    for ch = 1:num_channels
        [S,F,~] = stft(eeg_ica(ch,:), fs, ...
            'Window', hamming(128), 'OverlapLength', 64, 'FFTLength', 256);
        P = abs(S).^2;
        freq_power = mean(P,2);

        for b = 1:n_bands
            f1 = bands{b,2}; f2 = bands{b,3};
            idx_band = find(F >= f1 & F < f2);
            stft_energy_median(ch,b) = sum(freq_power(idx_band));
        end
    end

    % Wavelet energy analysis
    for ch = 1:num_channels
        [c,l] = wavedec(eeg_ica(ch,:), 5, 'db4');
        A5 = appcoef(c,l,'db4',5);
        D5 = detcoef(c,l,5); D4 = detcoef(c,l,4);
        D3 = detcoef(c,l,3); D2 = detcoef(c,l,2); D1 = detcoef(c,l,1);

        band_energy = zeros(1,n_bands);
        band_energy(1) = sum(A5.^2);
        band_energy(2) = sum(D5.^2);
        band_energy(3) = sum(D4.^2);
        band_energy(4) = sum(D3.^2);
        band_energy(5) = sum(D2.^2) + sum(D1.^2);

        wavelet_energy_median(ch,:) = band_energy;
    end

    %% Z-score normalization for highlighting outliers
    stft_zscore_median = (stft_energy_median - mean(stft_energy_median(:))) ./ std(stft_energy_median(:));
    wavelet_zscore_median = (wavelet_energy_median - mean(wavelet_energy_median(:))) ./ std(wavelet_energy_median(:));
end

% Normalize energy to ratio
stft_ratio = stft_energy_median ./ sum(stft_energy_median, 2);
wavelet_ratio = wavelet_energy_median ./ sum(wavelet_energy_median, 2);
resultstft = stft_ratio';
resultstftW = wavelet_ratio';

% Display results
fprintf('\nSTFT Energy Ratio:\n');
disp(array2table(stft_ratio, 'VariableNames', bands(:,1), 'RowNames', channel_names))

fprintf('\nWavelet Energy Ratio:\n');
disp(array2table(wavelet_ratio, 'VariableNames', bands(:,1), 'RowNames', channel_names))

%% Compute intra-band channel variance (stability analysis)
stft_band_mse = zeros(1, n_bands);
for b = 1:n_bands
    band_data = stft_energy_median(:, b);
    stft_band_mse(b) = mean((band_data - mean(band_data)).^2);
end

wavelet_band_mse = zeros(1, n_bands);
for b = 1:n_bands
    band_data = wavelet_energy_median(:, b);
    wavelet_band_mse(b) = mean((band_data - mean(band_data)).^2);
end

% Display MSE results
fprintf('\n=== Intra-band channel variance (MSE) ===\n');
result_table = table(... 
    stft_band_mse', wavelet_band_mse', ...
    'VariableNames', {'STFT_MSE', 'Wavelet_MSE'}, ...
    'RowNames', bands(:,1));
disp(result_table);

%% Export key parameters (can save as CSV)
final_results = table(... 
    bands(:,1), stft_band_mse', wavelet_band_mse', ...
    'VariableNames', {'Band', 'STFT_MSE', 'Wavelet_MSE'});
