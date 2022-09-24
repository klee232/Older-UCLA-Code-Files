function weightcc = LPCC(frames, fs, bias, method)
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
%     weightcc: matrix, each row is a frame,
%               each column is a weighted lpcc

addpath '/home/ychung0410/Desktop/voicebox/sap-voicebox-master/voicebox';

% first calculate LPC
lpcs = LPC(frames, fs, bias, method);

% LPC to LPCC
[frame_num, order] = size(lpcs);
lpccs = zeros(frame_num, order);
for i = 1:1:frame_num
    lpccs(i, 1) = log(order+1);
    for j = 2:1:order
        c_sum = 0;
        for k = 1:1:j-1
           c_sum = c_sum + (k/j) * lpccs(i, k) * lpcs(i, j-k);
        end
        lpccs(i, j) = lpcs(i, j) + c_sum;
    end
end

% Parameter Weighting
wc = zeros(1, order);
for i= 1:1:order
        wc(i) = 1 + 13 / 2 * sin(pi * i / 13);
end
W = ones(frame_num, 1) * wc;
weightcc = lpccs .* W;
 
end