function hamming_frames = hamming_window(frames)
% Hamming window each frame, each row is a frame
% INPUT:
%     frames: matrix, each row is a frame,
%             number of column is the window length
% OUTPUT:
%     hamming_frames: matrix, each row is a frame

[frame_num, win_length] = size(frames);
window = hamming(win_length)' .* ones(frame_num, 1);
hamming_frames = frames .* window;

end