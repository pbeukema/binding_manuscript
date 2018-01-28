function surface_searchlight(SID);
%h = waitbar(0,'Initializing waitbar...');
% function gen_searchlight(spm_file, mask_file, varargin);
%remember that if files have been copied over from other computer, need to
%change file names with spm_chanspm_changepath('SPM.mat','/data/r2d4/subjects/','/home/pbeukema/r2d4/subjects/');
%e.g. spm_changepath('SPM.mat','/data/r2d4/subjects/','/home/pbeukema/r2d4/subjects/');

matlabpool open;
addpath('/home/pbeukema/r2d4/bin/');
addpath('/home/pbeukema/modMap/bin/');
addpath('/home/pbeukema/r2d4/subjects/');
addpath('/home/matlab/8.1/toolbox/spm/spm8');

%h = waitbar(0,'Initializing waitbar...');
% function gen_searchlight(spm_file, mask_file, varargin);


% General searchlight routine. Built for RSA
plot_flag = 0; % don't plot on each run
fold_size = 5;
outfname = 'ss';

SID = sprintf('0%s', num2str(SID));
% Load the SPM objects
spm_file1 = sprintf('/home/pbeukema/r2d4/subjects/%s_1/effectorGLM/SPM.mat', SID);
spm_file2 = sprintf('/home/pbeukema/r2d4/subjects/%s_2/effectorGLM/SPM.mat', SID);
mask_file = sprintf('/home/pbeukema/r2d4/subjects/%s_1/effectorGLM/mask.hdr', SID);


SPM_pre = load(spm_file1);
SPM_post = load(spm_file2);
SPM_pre = SPM_pre.SPM;
SPM_post = SPM_post.SPM;
% Load the mask
Vmask = spm_vol(mask_file);
Ymask = spm_read_vols(Vmask);
voxels = load(sprintf('/home/pbeukema/r2d4/subjects/%s_1/surface_searchlight/surface_voxels.mat', SID));

%Clear those serchlights that are empty:
vox_centers = voxels.centerindxs;
surface_voxels = voxels.n2v;
non_empty = find(~cellfun(@isempty,surface_voxels));
vox_centers = vox_centers(non_empty);

is_empty = find(cellfun(@isempty,surface_voxels));
surface_voxels(:,is_empty) = [];

% get only masked voxels as your seeds
index = find(Ymask(:));

% Setup a new output volumes

h_vol_change = NaN(size(Ymask));
h_vol_pre = NaN(size(Ymask));
h_vol_post = NaN(size(Ymask));


n_betas = length(SPM_pre.Sess(1).col);

%Preallocate for parfor loop.
pre = [];
post = [];
change = [];
tic;
parfor i = 1:numel(vox_centers);

    %Select the center voxel and searchlight voxels
    this_center = vox_centers(i);
    this_searchlight = surface_voxels{i};

    [x, y, z] = ind2sub(size(Ymask),this_searchlight);
    coord = double([x;y;z]); %searchlight voxels

    % Extract data
    y_data_pre = spm_get_data(SPM_pre.xY.VY, coord);
    y_data_post = spm_get_data(SPM_post.xY.VY, coord);

    %Remove columns of zeros o/w matrix may not be positive semi definite:
    y_data_pre(:, find(sum(abs(y_data_pre)) == 0)) = [];
    y_data_post(:, find(sum(abs(y_data_post)) == 0)) = [];

    %Occasionaly grabs data outside mask - unclear why
    if numel(y_data_pre)==0;
        pre(i) = NaN;
        post(i) = NaN;
        change(i) = NaN;
        continue
    end;

    % Orthogonalize y_data
    y_data_pre = grab95pca(y_data_pre);
    y_data_post = grab95pca(y_data_post);

    % Prewhiten the beta coefficients
    [beta_w_pre, beta_pre, resMS_pre] = mva_prewhiten_beta(y_data_pre, SPM_pre);
    [beta_w_post, beta_post, resMS_post] = mva_prewhiten_beta(y_data_post, SPM_post);

    % Run RSA to obtain distances between patterns
    [preH] = rsa_evalCorrectTrials(beta_w_pre, n_betas, plot_flag, fold_size);
    pre(i) = preH*100; %previously, x y z
    [postH] = rsa_evalCorrectTrials(beta_w_post, n_betas, plot_flag, fold_size);
    post(i) = postH*100;

    %Compute the difference between pre and post
    true_D = postH-preH;
    change(i) = true_D*100;

end;
toc;
h_vol_pre(vox_centers) = pre;
h_vol_post(vox_centers) = post;
h_vol_change(vox_centers) = change;

% Now save the output files;
[fp, fn, fe] = fileparts(Vmask.fname);

Vh = Vmask;
Vh.dt(1)= 16;
Vh.fname = fullfile(fp,sprintf('%s_H_change.img', outfname));
spm_write_vol(Vh, h_vol_change);

Vpre = Vmask;
Vpre.dt(1)= 16;
Vpre.fname = fullfile(fp,sprintf('%s_H_pre.img', outfname));
spm_write_vol(Vpre, h_vol_pre);

Vpost = Vmask;
Vpost.dt(1)= 16;
Vpost.fname = fullfile(fp,sprintf('%s_H_post.img', outfname));
spm_write_vol(Vpost, h_vol_post);
matlabpool close;
exit;S
