
function [H, dhat] = rsa_evalCorrectTrials(beta_w, n_betas)

%function [H, dhat] = rsa_evalCorrectTrials(y_raw, SPM, plot_flag, fold_size)

i=0;
for j=[1:n_betas:size(beta_w,1)];
    i= i+1;
    nU(:,:,i) = normr(beta_w(j:j+3,:));
end;


% Find the squared distance between each finger for each permutations 
n_voxels = size(y_data_pre,2);
num_pairs = combnk(1:size(nU,3),2);
for pair = 1:size(num_pairs,1);
    for finger1 = 1:size(nU,1);
        for finger2 = 1:size(nU,1);
            x = num_pairs(pair,:);
            foldm = (nU(finger1,:,x(1)) - nU(finger2,:,x(1)));
            foldl = (nU(finger1,:,x(2)) - nU(finger2,:,x(2)));
            d(finger1,finger2,pair) = (1/n_voxels)*foldm*foldl';
        end
    end
end

dhat = sum(d,3)./size(d,3);
K = length(dhat);
H = nansum(dhat(:))/(K*(K-1));

