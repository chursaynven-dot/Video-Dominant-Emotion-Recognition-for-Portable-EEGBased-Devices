%% EEG Signal Processing and Frequency Band Analysis (GPU Accelerated)
clear; close all; clc;
% Enable GPU computation if available
if canUseGPU()
    disp('GPU available, enabling GPU acceleration');
    useGPU = true;
else
    disp('GPU not available, using CPU');
    useGPU = false;
end

%% 1. Parameters and Data Loading
fs = 128; % Sampling rate
gammahigh = 60;
num_channels = 11; % 11 channels including O1/O2
% Define frequency bands
bands = {
    'delta', 1, 4;
    'theta', 4, 8;
    'alpha', 8, 14;
    'beta',  14, 30;
    'gamma', 30, gammahigh;
};
n_bands = size(bands, 1);
% Sliding window parameters
window_size = 2 * fs; % 2-second window
overlap = 0.5; % 50% overlap
step_size = round(window_size * (1 - overlap));
% Channel names
channel_names = {'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'T7', 'T8', 'Pz', 'O1', 'O2'};
% Load all .mat files
data_folder = '\Video_e_g\Emo1\'; 
file_list = dir(fullfile(data_folder, '*.mat'));
num_files = length(file_list);
% Preallocate feature matrix
num_features = 151;
features = zeros(num_files, num_features);

%% 2. Loop through all files
for file_idx = 1:num_files
    file_name = file_list(file_idx).name;
    fprintf('Processing file %d/%d: %s...\n', file_idx, num_files, file_name);  
    % Load EEG data
    data = load(fullfile(data_folder, file_name));
    eeg_data = data.data_use_decimated; % Adjust variable name as needed    
    % Check channel count
    if size(eeg_data, 1) ~= num_channels
        error('Channel count mismatch in file %s', file_name);
    end    
    %% 3. Preprocessing (GPU accelerated)
    if useGPU
        eeg_data = gpuArray(eeg_data);
    end
    
    eeg_data = eeg_data - mean(eeg_data, 2); % Remove DC offset
    
    % Bandpass filter (1-60 Hz)
    [b, a] = butter(4, [1 gammahigh]/(fs/2), 'bandpass');
    if useGPU
        b = gpuArray(b); a = gpuArray(a);
    end
    eeg_filtered = filtfilt(b, a, eeg_data')';
    
    % Notch filter at 50 Hz
    wo = 50/(fs/2); bw = wo/35;
    [b, a] = iirnotch(wo, bw);
    if useGPU
        b = gpuArray(b); a = gpuArray(a);
    end
    eeg_filtered = filtfilt(b, a, eeg_filtered')';
    
    % Common average reference (CAR)
    avg_ref = mean(eeg_filtered, 1);
    eeg_car = eeg_filtered - avg_ref;
    
    %% 4. ICA for artifact removal
    eeg_car_cpu = gather(eeg_car); % FastICA requires CPU
    
    addpath('./FastICA_2.5/FastICA_25/'); % relative path to FastICA
    tic;
    [icasig, A, W] = fastica(eeg_car_cpu, 'approach', 'symm', 'g', 'tanh', ...
                            'lastEig', num_channels, 'numOfIC', num_channels, ...
                            'stabilization', 'on', 'verbose', 'off');
    toc;    
    kurt_values = kurtosis(icasig, 1, 2);
    artifact_components = find(abs(kurt_values) > 5);
    icasig(artifact_components,:) = 0;
    eeg_ica = A * icasig;    
    if useGPU
        eeg_ica = gpuArray(eeg_ica);
    end
    
    %% 5. Wavelet denoising
    eeg_denoised = zeros(size(eeg_ica), 'like', eeg_ica);
    if useGPU
        for ch = 1:num_channels
            signal = eeg_ica(ch,:);
            eeg_denoised(ch,:) = wdenoise(signal, 5, 'Wavelet', 'db4', ...
                                         'DenoisingMethod', 'SURE', ...
                                         'ThresholdRule', 'Soft', ...
                                         'NoiseEstimate', 'LevelIndependent');
        end
    else
        for ch = 1:num_channels
            eeg_denoised(ch,:) = wden(eeg_ica(ch,:), 'rigrsure', 's', 'sln', 5, 'db4');
        end
    end
    
    %% 6. Feature extraction
    fb = cwtfilterbank('SignalLength', size(eeg_denoised, 2), ...
                      'SamplingFrequency', fs, ...
                      'Wavelet', 'morse', ...
                      'FrequencyLimits', [1 gammahigh], ...
                      'VoicesPerOctave', 10);    
    % Channel indices
    fp1_idx = find(strcmp(channel_names, 'FP1'));
    fp2_idx = find(strcmp(channel_names, 'FP2'));
    f3_idx = find(strcmp(channel_names, 'F3'));
    f4_idx = find(strcmp(channel_names, 'F4'));
    c4_idx = find(strcmp(channel_names, 'C4'));
    t8_idx = find(strcmp(channel_names, 'T8'));
    o1_idx = find(strcmp(channel_names, 'O1'));
    o2_idx = find(strcmp(channel_names, 'O2'));    
    % Bandpass filters for each frequency band
    band_filters = cell(n_bands, 1);
    for b = 1:n_bands
        [band_filters{b}.b, band_filters{b}.a] = butter(4, [bands{b,2} bands{b,3}]/(fs/2), 'bandpass');
        if useGPU
            band_filters{b}.b = gpuArray(band_filters{b}.b);
            band_filters{b}.a = gpuArray(band_filters{b}.a);
        end
    end
    
    % Wavelet energy and band signals
    power_results.wavelet = zeros(num_channels, n_bands, 'like', eeg_denoised);
    band_signals = cell(num_channels, n_bands);
    
    for ch = 1:num_channels
        signal = eeg_denoised(ch, :);
        [cfs, frq] = wt(fb, signal);
        valid_idx = (frq >= 1 & frq < gammahigh);
        total_energy = sum(abs(cfs(valid_idx,:)).^2, 'all');
        
        for b = 1:n_bands
            band_idx = (frq >= bands{b,2}) & (frq < bands{b,3});
            power_results.wavelet(ch,b) = sum(abs(cfs(band_idx,:)).^2, 'all') / total_energy * 100;
            
            if useGPU
                band_signal = filtfilt(band_filters{b}.b, band_filters{b}.a, signal')';
            else
                band_signal = filtfilt(band_filters{b}.b, band_filters{b}.a, gather(signal)')';
            end
            band_signals{ch,b} = band_signal;
        end
    end
    
    % F3-C4 coherence
    f3_c4_coherence = zeros(1, n_bands);
    for b = 1:n_bands
        sig1 = band_signals{f3_idx, b};
        sig2 = band_signals{c4_idx, b};
        [coh, ~] = mscohere(sig1, sig2, hamming(128), 64, 128, fs);
        f3_c4_coherence(b) = mean(coh);
    end
    
    % Extract alpha/beta/gamma features
    alpha_idx = find(strcmp(bands(:,1), 'alpha'));
    beta_idx  = find(strcmp(bands(:,1), 'beta'));
    gamma_idx = find(strcmp(bands(:,1), 'gamma'));
    
    fp1_alpha = band_signals{fp1_idx, alpha_idx};
    fp1_alpha_median = median(fp1_alpha);
    fp1_alpha_peak   = max(abs(fp1_alpha));
    
    fp2_alpha = band_signals{fp2_idx, alpha_idx};
    fp2_alpha_median = median(fp2_alpha);
    fp2_alpha_peak   = max(abs(fp2_alpha));
    
    f3_alpha = band_signals{f3_idx, alpha_idx};
    f3_alpha_median = median(f3_alpha);
    f3_alpha_peak   = max(abs(f3_alpha));
    
    f3_beta = band_signals{f3_idx, beta_idx};
    f3_beta_median = median(f3_beta);
    f3_beta_peak   = max(abs(f3_beta));
    
    f4_alpha = band_signals{f4_idx, alpha_idx};
    f4_alpha_median = median(f4_alpha);
    f4_alpha_peak   = max(abs(f4_alpha));
    
    f4_beta = band_signals{f4_idx, beta_idx};
    f4_beta_median = median(f4_beta);
    f4_beta_peak   = max(abs(f4_beta));
    
    o1_beta = band_signals{o1_idx, beta_idx};
    o1_beta_median = median(o1_beta);
    o1_beta_peak   = max(abs(o1_beta));
    
    t8_gamma = band_signals{t8_idx, gamma_idx};
    t8_gamma_median = median(t8_gamma);
    t8_gamma_peak   = max(abs(t8_gamma));
    
    f3_wavelet = power_results.wavelet(f3_idx, :);
    c4_wavelet = power_results.wavelet(c4_idx, :);
    o1_wavelet = power_results.wavelet(o1_idx, :);
    o2_wavelet = power_results.wavelet(o2_idx, :);
    
    %% 7. PSD Sliding Window Analysis
    psd_features = zeros(1, 110);
    psd_counter = 1;
    
    for ch = 1:num_channels
        signal = eeg_denoised(ch, :);
        num_samples = length(signal);
        num_windows = floor((num_samples - window_size) / step_size) + 1;
        
        band_psd_values = zeros(n_bands, num_windows);
        
        for w = 1:num_windows
            start_idx = (w-1)*step_size + 1;
            end_idx   = start_idx + window_size - 1;
            window_data = signal(start_idx:end_idx);
            [psd, freq] = pwelch(window_data, hamming(window_size), [], [], fs);
            
            for b = 1:n_bands
                band_mask = (freq >= bands{b,2}) & (freq < bands{b,3});
                band_psd_values(b, w) = mean(psd(band_mask));
            end
        end
        
        for b = 1:n_bands
            psd_features(psd_counter)   = mean(band_psd_values(b, :));
            psd_features(psd_counter+1) = max(band_psd_values(b, :));
            psd_counter = psd_counter + 2;
        end
    end
    
    % Gather results from GPU
    if useGPU
        f3_c4_coherence = gather(f3_c4_coherence);
        f3_wavelet = gather(f3_wavelet);
        c4_wavelet = gather(c4_wavelet);
        o1_wavelet = gather(o1_wavelet);
        o2_wavelet = gather(o2_wavelet);
        fp1_alpha_median = gather(fp1_alpha_median);
        fp1_alpha_peak   = gather(fp1_alpha_peak);
        fp2_alpha_median = gather(fp2_alpha_median);
        fp2_alpha_peak   = gather(fp2_alpha_peak);
        f3_alpha_median  = gather(f3_alpha_median);
        f3_alpha_peak    = gather(f3_alpha_peak);
        f3_beta_median   = gather(f3_beta_median);
        f3_beta_peak     = gather(f3_beta_peak);
        f4_alpha_median  = gather(f4_alpha_median);
        f4_alpha_peak    = gather(f4_alpha_peak);
        f4_beta_median   = gather(f4_beta_median);
        f4_beta_peak     = gather(f4_beta_peak);
        o1_beta_median   = gather(o1_beta_median);
        o1_beta_peak     = gather(o1_beta_peak);
        t8_gamma_median  = gather(t8_gamma_median);
        t8_gamma_peak    = gather(t8_gamma_peak);
        psd_features     = gather(psd_features);
    end
    
    % Combine all features
    features(file_idx, :) = [
        f3_c4_coherence, ...
        f3_wavelet, ...
        c4_wavelet, ...
        o1_wavelet, ...
        o2_wavelet, ...
        fp1_alpha_median, fp1_alpha_peak, ...
        fp2_alpha_median, fp2_alpha_peak, ...
        f3_alpha_median, f3_alpha_peak, ...
        f3_beta_median, f3_beta_peak, ...
        f4_alpha_median, f4_alpha_peak, ...
        f4_beta_median, f4_beta_peak, ...
        o1_beta_median, o1_beta_peak, ...
        t8_gamma_median, t8_gamma_peak, ...
        psd_features
    ];
end

%% 8. Save results
disp('Feature extraction completed');

% Generate PSD feature names
psd_feature_names = {};
for ch = 1:num_channels
    for b = 1:n_bands
        band_name = bands{b,1};
        ch_name = channel_names{ch};
        psd_feature_names{end+1} = sprintf('%s %s PSD mean', ch_name, band_name);
        psd_feature_names{end+1} = sprintf('%s %s PSD max', ch_name, band_name);
    end
end

% Combine all feature names
feature_names = [
    {'F3-C4 delta coherence', 'F3-C4 theta coherence', 'F3-C4 alpha coherence', ...
     'F3-C4 beta coherence', 'F3-C4 gamma coherence', ...
     'F3 delta wavelet', 'F3 theta wavelet', 'F3 alpha wavelet', 'F3 beta wavelet', 'F3 gamma wavelet', ...
     'C4 delta wavelet', 'C4 theta wavelet', 'C4 alpha wavelet', 'C4 beta wavelet', 'C4 gamma wavelet', ...
     'O1 delta wavelet', 'O1 theta wavelet', 'O1 alpha wavelet', 'O1 beta wavelet', 'O1 gamma wavelet', ...
     'O2 delta wavelet', 'O2 theta wavelet', 'O2 alpha wavelet', 'O2 beta wavelet', 'O2 gamma wavelet', ...
     'FP1 alpha median', 'FP1 alpha peak', ...
     'FP2 alpha median', 'FP2 alpha peak', ...
     'F3 alpha median', 'F3 alpha peak', ...
     'F3 beta median', 'F3 beta peak', ...
     'F4 alpha median', 'F4 alpha peak', ...
     'F4 beta median', 'F4 beta peak', ...
     'O1 beta median', 'O1 beta peak', ...
     'T8 gamma median', 'T8 gamma peak'}, ...
     psd_feature_names
];

disp('Feature list:');
disp(feature_names');
