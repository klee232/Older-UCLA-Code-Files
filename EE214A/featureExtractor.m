%% Reset workspace
clear; clc; close all;
rng(1);

% Parameters
allFiles = 'allFiles.txt';
speech_likelihood_thresh = 0.7; % Default 0.7
voiced_likelihood_thresh = 0.7; % Default 0.7

%% Extract features
newFeatureDict = containers.Map;
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
for cnt = 1:length(myFiles)
    [snd,fs] = audioread(myFiles{cnt});
    
    % Compute speech likelyhood vector
    speachlik= v_vadsohn(snd,fs,'nb');
    
    % Compute fist 12 MFCCs. Note all parametetes are the default except
    % the last one so that there could be a matach between vector lengths.
    c = v_melcepst(snd,fs,'M',12,floor(3*log(fs)),0.02*fs);
    % Filter using threshold
    c = c(speachlik(:,3) > speech_likelihood_thresh,:);
    % Remove time dimension
    c = [mean(c)'; std(c)'];
    
    % Compute pitch
    [pitch,~,pv] = v_fxpefac(snd,fs);
    % Filter using threshold
    pitch = pitch(pv > voiced_likelihood_thresh);
    % Remove time dimension
    pitch = [mean(pitch)'; std(pitch)'];

    % Compute harmonic ratio
    hr = harmonicRatio(snd,fs);
    % Filter using voice threshold
    hr = hr(pv > voiced_likelihood_thresh);
    % Remove time dimension
    hr = [mean(hr)'; std(hr)'];
    
    % Save pitch and MFCC features in vector 
    feature = [hr; pitch; c];
    % Store in map
    newFeatureDict(myFiles{cnt}) = feature;

    if(mod(cnt,1)==0)
        disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
    end
end

% Save feature dict
save('newFeatureDict_with_hr','newFeatureDict');