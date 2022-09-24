
%##############################################################
% Sample script to perform short utterance speaker verficiation
% ECE214A: Digital Speech Processing, Winter 2020
%##############################################################

%% Reset workspace
clear;
clc;
close all;
rng(1);

%%
% Define lists
allFiles = 'allFiles.txt';
trainList = 'train_read_trials.txt';  
testList = 'test_read_trials.txt';
%#################################################################################################
% tic
% %
% % Extract features
%load('featureDict.mat');
%#################################################################################################
fid = fopen(allFiles);
myData = textscan(fid,'%s');
fclose(fid);
myFiles = myData{1};
%##################################################################################################
% for cnt = 1:length(myFiles)
%     [snd,fs] = audioread(myFiles{cnt});
%     try
%         %[F0,lik] = fast_mbsc_fixedWinlen_tracking(snd,fs);
%         %featureDict(myFiles{cnt}) = mean(F0(lik>0.45));
%         c = v_melcepst(snd,fs,'dD');
%         feature = [mean(c)'; std(c)'];
%         featureDict(myFiles{cnt}) = [featureDict(myFiles{cnt}); feature];
%     catch
%         disp(["No features for the file ", myFiles{cnt}]);
%     end
%     
%     if(mod(cnt,1)==0)
%         disp(['Completed ',num2str(cnt),' of ',num2str(length(myFiles)),' files.']);
%     end
% end
% %save('featureDict');

%% Train the classifier
load('newFeatureDict');
load('lists_labels.mat');
featureDict = newFeatureDict;
[List1,List2,List3,List4,Labels1,Labels2] = decidelist(1,1);
% ################################################################################################
% fid = fopen(trainList,'r');
% myData = textscan(fid,'%s %s %f');
% fclose(fid);
% fileList1 = myData{1};
% fileList2 = myData{2};
% trainLabels = myData{3};
% ################################################################################################
trainFeatures = zeros(length(Labels1),length(newFeatureDict(List1{1})));
for cnt = 1:length(Labels1)
    trainFeatures(cnt,:) = abs(newFeatureDict(List1{cnt})-newFeatureDict(List2{cnt})); 
end

Mdl = fitcknn(trainFeatures,Labels1,'NumNeighbors',2300,'DistanceWeight','inverse','Standardize',1);
%##################################################################################################
% tried optimizer (even worse)
%Mdl = fitcknn(trainFeatures,Labels1,'OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',...
%    struct('AcquisitionFunctionName','expected-improvement-plus'))

% #################################################################################################
% prior_1 = sum(trainLabels)/length(trainLabels);
% prior_0 = 1 - prior_1;
% cost = [0 1/prior_0; 1/prior_1 0];
% 
% Mdl = fitcknn(trainFeatures,trainLabels,...
%    'OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',...
%    struct('AcquisitionFunctionName','expected-improvement-plus'),...
%    'Cost',cost,'Standardize',1);

%% Test the classifier
% fid = fopen(testList);
% myData = textscan(fid,'%s %s %f');
% fclose(fid);
% fileList1 = myData{1};
% fileList2 = myData{2};
% testLabels = myData{3};
% ###########################################################################################
testFeatures = zeros(length(Labels2),length(newFeatureDict(List3{1})));
for cnt = 1:length(Labels2)
    testFeatures(cnt,:) = abs(newFeatureDict(List3{cnt})-newFeatureDict(List4{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, Labels2);
disp(['The EER is ',num2str(eer),'%.']);
function [List1,List2,List3,List4,Labels1,Labels2] = decidelist(train_p,test_p)
load('lists_labels.mat');
if(train_p==1)
    List1 = readtrainlist1;
    List2 = readtrainlist2;
    Labels1 = readtrainlabel;
elseif(train_p==2)
    List1 = phonetrainlist1;
    List2 = phonetrainlist2;
    Labels1 = phonetrainlabel;
else
    List1 = mixtrainlist1;
    List2 = mixtrainlist2;
    Labels1 = mixtrainlabel;
end
if(test_p==1)
    List3 = readtestlist1;
    List4 = readtestlist2;
    Labels2 = readtestlabel;
elseif(test_p==2)
    List3 = phonetestlist1;
    List4 = phonetestlist2;
    Labels2 = phonetestlabel;
else
    List3 = mixtestlist1;
    List4 = mixtestlist2;
    Labels2 = mixtestlabel;
end
end
%%toc
%%