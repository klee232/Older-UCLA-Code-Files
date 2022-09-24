function [feature_vec] = eval_feat(features,labels)
%EVAL_FEAT Evaluates the features based on high between-speaker variability
% and low within-speaker variability

norm_feat = normalize(features);
feature_vec = mean(abs(norm_feat(labels==0,:))) - mean(abs(norm_feat(labels==1,:)));
end

