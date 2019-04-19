function  data = representational_distances()
% extracts raw data and distances from each ROI and subject and computes
% representational distances between fingers and writes csv summarizing
% data

% list of subjects
all_subs = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18'};

% list of ROIs
rois = {'m1', 'S1', 'pmd', 'pmv', 'sma', 'spl'};

% initialize output csv
data = [];
fig5_mean_distance = [];
fig5_median_split_distance = [];

for roi = 1:length(rois);
    
    % initialize summmary tables
    table3_trained = []; 
    table4_trained = [];
    table5_trained = [];
    table3_control = [];
    table4_control = [];
    table5_control = [];

    region = rois{roi};
    region = sprintf('%s%s',region, '.nii'); 
    
    s = struct;
    for i = 1:length(all_subs)
        SID = all_subs{i}
        spm_file_pre = sprintf('/data/r2d4/subjects/%s_1/complexGLM/SPM.mat', SID);
        spm_file_post = sprintf('/data/r2d4/subjects/%s_2/complexGLM/SPM.mat', SID);
        mask_file = sprintf('/data/r2d4/subjects/%s_1/rois/%s',SID, region);
        masked_img = load_untouch_nii(mask_file);

        SPM_pre = load(spm_file_pre);
        SPM_post = load(spm_file_post);
        SPM_pre = SPM_pre.SPM;
        SPM_post = SPM_post.SPM;

        Vmask = spm_vol(mask_file);
        roiMask = spm_read_vols(Vmask);
        n_betas = length(SPM_pre.Sess(1).col);
    
        roi_ind = find(roiMask);
        [x, y, z] = ind2sub(size(roiMask),roi_ind);
        vox = [x y z]'; 

        y_data_pre = spm_get_data(SPM_pre.xY.VY,vox);
        y_data_post =  spm_get_data(SPM_post.xY.VY,vox);
        y_data_pre = grab95pca(y_data_pre);
        y_data_post = grab95pca(y_data_post);

        % Prewhiten the beta coefficients 
        [beta_w_pre, beta_pre, resMS_pre] = mva_prewhiten_beta(y_data_pre, SPM_pre);
        [beta_w_post, beta_post, resMS_post] = mva_prewhiten_beta(y_data_post, SPM_post);

        % Run RSA to obtain distance btw fingers before and after training
        [preH_true, dhatpre] = rsa(beta_w_pre, n_betas);
        [postH_true, dhatpost] = rsa(beta_w_post, n_betas);
      
        true_D = postH_true-preH_true;
        rdmspre(:,:,i) = dhatpre;
        rdmspost(:,:,i) = dhatpost;
      
        data = [data, [str2num(SID);true_D; postH_true;preH_true]];
     
        % Compute the distances between the bound and unbound pairs
        bound_vals = [0 1 1 0; 0 0 0 1; 0 0 0 1; 0 0 0 0];
        unbound_vals = [0 0 0 1; 0 0 1 0; 0 0 0 0; 0 0 0 0];
        dhatpre_bound = nanmean(dhatpre(find(bound_vals)));
        dhatpre_unbound =  nanmean(dhatpre(find(unbound_vals)));
        dhatpost_bound = nanmean(dhatpost(find(bound_vals)));
        dhatpost_unbound = nanmean(dhatpost(find(unbound_vals)));        

        % get the group identifier:
        switch SID;
            case {'S1','S2', 'S3', 'S4', 'S5', 'S6', 'S13', 'S14', 'S15'}
                group_identifier = 1;
                table3_trained = [table3_trained; preH_true, postH_true];
                table4_trained = [table4_trained; dhatpre_bound, dhatpost_bound];
                table5_trained = [table5_trained; dhatpre_unbound, dhatpost_unbound];
            case {'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S16', 'S17', 'S18'};
                group_identifier = 2;
                table3_control = [table3_control; preH_true, postH_true];
                table4_control = [table4_control; dhatpre_bound, dhatpost_bound];
                table5_control = [table5_control; dhatpre_unbound, dhatpost_unbound];
        end;
        fig5_mean_distance = [fig5_mean_distance ; preH_true, group_identifier, roi, 1]; 
        fig5_mean_distance = [fig5_mean_distance ; postH_true, group_identifier, roi, 2]; 

        fig5_median_split_distance = [fig5_median_split_distance ; dhatpre_bound, group_identifier, roi, 1, 1]; 
        fig5_median_split_distance = [fig5_median_split_distance ; dhatpost_bound, group_identifier, roi, 2, 1]; 
        fig5_median_split_distance = [fig5_median_split_distance ; dhatpre_unbound, group_identifier, roi, 1, 2]; 
        fig5_median_split_distance = [fig5_median_split_distance ; dhatpost_unbound, group_identifier, roi, 2, 2];


        csvwrite('~/Desktop/all_voxels/fig5_mean_distance.csv', fig5_mean_distance)
        csvwrite('~/Desktop/all_voxels/fig5_median_split_distance.csv', fig5_median_split_distance)

        s(i).subject_number = i;
        s(i).y_raw_pre = beta_w_pre;
        s(i).y_raw_post = beta_w_post;
    end
    
    save(sprintf('/data/r2d4/%s_raw_data.mat',rois{roi}), 's');
    sprintf('processed %s',region)
    csvwrite(sprintf('~/Desktop/all_voxels/table_3_%s.csv', region), [table3_trained, table3_control]);
    csvwrite(sprintf('~/Desktop/Desktop/all_voxels/table_4_%s.csv', region), [table4_trained, table4_control]);
    csvwrite(sprintf('~/Desktop/Desktop/all_voxels/table_5_%s.csv', region), [table5_trained, table5_control]);
end

