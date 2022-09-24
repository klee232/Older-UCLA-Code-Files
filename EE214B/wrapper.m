clear;clc;


%%Please download voicebox and add it to your path
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% EDIT THIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
noisy = 0; %0 for clean and 1 for noisy
addpath '/home/ychung0410/Desktop/voicebox/sap-voicebox-master/voicebox';
%%% add your voicebox path

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The wave files can be in different sub-folders but it will be good if you
% can give a parent folder with the all the folders of the wave files
Fs = 16000;  %sampling rate of TIMIT
curr_dir = pwd;

if noisy==0
    fileDir = [curr_dir '/']; %PARENT FOLDER
    scp_file =  'clean_file_list'; %list of all the files for feature extraction
    %%%%%%%%% EDIT THIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ftr_Dir = 'my_features/'; %location of the parent folder to store 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%the extracted features
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    fileDir = [curr_dir '/']; %PARENT FOLDER
    scp_file =  'noisy_file_list'; %list of all the files for feature extraction
    %%%%%%%%% EDIT THIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ftr_Dir = 'my_features/'; %location of the parent folder to store
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%the extracted features    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
fid=fopen(scp_file,'r');
count=1;
while ~feof(fid)
    
    tline = fgets(fid);
    temp_in = regexp(tline,'[\r\f\n]','split');
    temp = temp_in{1};
    filenames{count} = temp;
    count=count+1;
    
end
fclose(fid);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Setting Parameters
% flag_specsub = 0;
% flag_mfcc = 0;
% flag_lpc = 0;
flag_lpcc = 1;
% flag_cmnv = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for cnt = 1:1:length(filenames)
    
    fileName = filenames{cnt};
    snd_FilePath =  [fileDir fileName];
    fprintf('Processing %s\n', fileName);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%% Edit this part to extract your custom features %%%%%%%%%%%%%%%%%
    [rawdata, fsamp] = audioread(snd_FilePath);
    
%     if flag_specsub == 1
%         rawdata = v_specsub(rawdata, fsamp);
%     end
    
    frames = split_frame(rawdata, fsamp);
    
%     if flag_mfcc == 1
%         mfccs = MFCC(frames, fsamp, 'M');
%     end
    
    frames = hamming_window(frames);
    
%     if flag_lpc == 1
%         lpcs = LPC(frames, fsamp, 4, 'covar');
%     end
%     
    if flag_lpcc == 1
        lpccs = LPCC(frames, fsamp, 4, 'covar');
    end
%     if flag_cmnv == 1
%     end

    ftr = lpccs;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    inds = strfind(fileName,'/');
    dirstore = [curr_dir '/' ftr_Dir fileName(1:inds(end)-1)];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Windows users might need to edit this %%%%
    if(~exist(dirstore))
        system(['mkdir -p ' dirstore]);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dlmwrite([curr_dir '/' ftr_Dir fileName(1:end-4) '.txt'],ftr);
    
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
