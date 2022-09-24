function [X_reduced] = pca_dim_reduct(X,output_dim)
%PCA_DIM_REDUCT Summary of this function goes here
    [~,score] = pca(X);
    X_reduced = score(:,1:output_dim);
end

