function [X_reduced] = mrmr_dim_reduct(X,Y,output_dim)
%MRMR_DIM_REDUCT Summary of this function goes here
    idx = fscmrmr(X,Y);
    X_reduced = X(:,idx <= output_dim);
end

