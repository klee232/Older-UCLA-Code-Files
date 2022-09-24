function frames = split_frame(input, fs)
% input vector split into matrix, each row is a frame
% the rest of the signal that is less than one frame is discarded
% INPUT:
%     input:  vector, raw data
%     fs:     sampling rate
% OUTPUT:
%     frames: matrix, each row is a frame


win_time = 20 * 10^(-3);        % window time duration (20ms)
win_length = fs * win_time;     % window length (samples)
win_timestep = 10 * 10^(-3);    % window time step (10ms)
win_step = fs * win_timestep;   % window step (samples)

% number of frames
frame_num = floor((length(input)-win_length) / win_step) + 1;

frames = zeros(frame_num, win_length);
for i = 1:1:frame_num
    start = win_step * (i-1) + 1;
    stop = start + win_length - 1;
    frames(i, :) = input(start:stop);
end

end