clear;
load('newFeatureDict.mat');
for train_p = 1:3
    for test_p = 1:3
[List1,List2,List3,List4,Labels1,Labels2] = decidelist(train_p,test_p);
trainFeatures = zeros(length(Labels1),length(newFeatureDict(List1{1})));
for cnt = 1:length(Labels1)
    trainFeatures(cnt,:) = abs(newFeatureDict(List1{cnt})-newFeatureDict(List2{cnt}));
end
Mdl = fitcknn(trainFeatures,Labels1,'NumNeighbors',700,'DistanceWeight','inverse','Standardize',1);
testFeatures = zeros(length(Labels2),length(newFeatureDict(List3{1})));
for cnt = 1:length(Labels2)
    testFeatures(cnt,:) = abs(newFeatureDict(List3{cnt})-newFeatureDict(List4{cnt}));
end

[~,prediction,~] = predict(Mdl,testFeatures);
testScores = (prediction(:,2)./(prediction(:,1)+1e-15));
[eer,~] = compute_eer(testScores, Labels2);
disp(['The EER is ',num2str(eer),'% for list figuer = ',num2str(train_p),' and ',num2str(test_p)]);
    end
end
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