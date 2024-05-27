% This is the code for setting up workshop folder
% as part of the "Wave Your Way" workshop presented
% at the GEM 2024

% Run this file to check your environment


% Check permissions for Exercise1
fileName = 'ex0.m';
[fid,errmsg] = fopen(fileName, 'w');
if ~isempty(errmsg)&&strcmp(errmsg,'Permission denied') 
    fprintf('\nError: You do not have write permission in the folder containing (%s).\n',fileName);
    fprintf('\nPlease make a copy of the original workshop folder and navigate to it.\n');
    fprintf('You will run the exercises out of the repository folder you created.\n');
else
    fprintf('\nWelcome to the Wave-Your-Way Workshop at GEM 2024!\n');
    fprintf('\nYou have write permission in this folder.\n');
    fprintf('\nInitializing the exercises...\n');
    % Add files to path
    addpath(genpath(pwd)); 

    % addpath(fullfile(pwd,'utils'));
    % addpath(genpath('utils'));    
    % addpath(fullfile(pwd,'E1_DesignTrainModel'));
    % addpath(genpath('E1_DesignTrainModel'));
    % addpath(fullfile(pwd,'E2_TransferLearn'));
    % addpath(genpath('E2_TransferLearn'));
    % addpath(fullfile(pwd,'E3_BayesianOptimization'));
    % addpath(genpath('E3_BayesianOptimization'));
    % addpath(fullfile(pwd,'E4_ModelCompression'));
    % addpath(genpath('E4_ModelCompression'));
    % addpath(fullfile(pwd,'E5_ONNX'));
    % addpath(genpath('E5_ONNX'));
    % addpath(fullfile(pwd,'E6_DevCloud'));
    % addpath(genpath('E6_DevCloud'));
    % addpath(fullfile(pwd,'E7_CodeGeneration'));
    % addpath(genpath('E7_CodeGeneration'));
    % addpath(fullfile(pwd,'E8_Resources'));
    % addpath(genpath('E8_Resources'));
    % These lines initialize the Simulink model for Exercise 1
    %load_system('ObstacleAvoidanceDemo.slx');
    %bdclose('ObstacleAvoidanceDemo.slx');
    fprintf('\nEnvironment Check is Complete!\n');
    fprintf('\nEnjoy the workshop!\n');
end
