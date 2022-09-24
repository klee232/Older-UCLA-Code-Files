function lpcs = LPC(frames, fs, bias, method)
% using function from voicebox, find the lpc of each frame,
% with the autocorrelation or covariance method
% INPUT:
%     frames: matrix, each row is a frame
%     fs:     sampling rate
%     method: the method to compute lpc
%               'covar': using the covariance method
%               'auto': using the autocorrelation method
%     bias:   order bias, adding to the order p, usually 2~4 
% OUTPUT:
%     lpcs: matrix, each row is a frame, each column is a lpc

addpath '/home/ychung0410/Desktop/voicebox/sap-voicebox-master/voicebox';

[frame_num, ~] = size(frames);
p = 2 * (fs / 2) / 1000 + bias;          % the order of lpc, 2 * Fs(kHz) + 2~4
lpcs = zeros(frame_num, p); 

if strcmp(method, 'covar')
    for i = 1:1:frame_num
        lpc = v_lpccovar(frames(i, :), p);
        lpcs(i, :) = lpc(2:p+1);
    end
    
    
elseif strcmp(method, 'auto')
    for i = 1:1:frame_num
        lpc = v_lpcauto(frames(i, :), p);
        lpcs(i, :) = lpc(2:p+1);
    end
    
    
else
    disp('ERROR! Method not specified!');
    disp('lpcs = LPC(frames, fs, bias, method)');
    disp('Please specify the method ("cov" or "auto")!');
end


end