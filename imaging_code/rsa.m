
function [H, dhat] = rsa_evalCorrectTrials(beta_w, n_betas, plot_flag, fold_size)

%function [p_value, G, H, d, distance, U, frobDist, frobP] = rsa_evalCorrectTrials(y_raw, SPM, plot_flag, fold_size)
%
% This function returns the cross validated distance between the effectors
% from the raw time series and the SPM file. It relies on J. Diedrichsen's
% mva_prewhiten_beta routine.
% Written by P. Beukema 02/08/15
%
% Inputs
% y_raw : raw time series
% SPM = full spm file
% plot_flag : plots distance matrix and CDS matrix
% fold_size = size of folds eg num_runs-1 (here 6) for LVOCC
%
% Outputs
% p_value =  P(H^_<0)
% G = cross validated estimate of G
% H = [Hhat(1), ...Hhat(m)] : average squared distance for each fold.
% d = d^2 = [d_1^2, ..., d^2_m] : all distances across every pairing of m
% distance = (sum(d,3)./size(d,3)) used for searchlight
% U = full model (P voxels by k conditions)
% frobDist = frobenious distance between observed and expected matrix
% frobP = p value for frobDist


% Generate prewhitened betas from y_raw and SPM.mat

i=0;
for j=[1:n_betas:size(beta_w,1)];
    %First build matrix of U's which is k conditions, by P voxels by M # of U's
    i= i+1;
    %build matrix with normalized pattern.
    nU(:,:,i) = normr(beta_w(j:j+3,:));
end;

num_pairs = combnk(1:size(nU,3),2);
% Find the squared distance between each finger for each permutations
% This generates all distances across each fold
for pair = 1:size(num_pairs,1);
    for finger1 = 1:size(nU,1);
        for finger2 = 1:size(nU,1);
            x = num_pairs(pair,:);
            foldm = (nU(finger1,:,x(1)) - nU(finger2,:,x(1)));
            foldl = (nU(finger1,:,x(2)) - nU(finger2,:,x(2)));
            d(finger1,finger2,pair) = foldm*foldl';
        end
    end
end

%Return H
dhat = sum(d,3)./size(d,3);
K = length(dhat);
dhat(dhat==Inf)=NaN;
H = nansum(dhat(:))/(K*(K-1));
