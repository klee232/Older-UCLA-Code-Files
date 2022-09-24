function result = table_test(Mdl,label,list1,list2)
load('newFeatureDict.mat');
testFeatures = zeros(length(label),length(newFeatureDict(list1{1})));
for cnt = 1:length(label)
    testFeatures(cnt,:) = abs(newFeatureDict(list1{cnt})-newFeatureDict(list2{cnt}));
end
[label_predict,~,~] = predict(Mdl,testFeatures);
table = zeros(2,length(label));
for i = 1:length(label)
    str = string(list1(i));
    person = sscanf(str,('WavData/phone/%3d%1a_0_phone_%2d.wav'));
    table(1,i) = person;
    table(2,i) = (1-abs(label(i)-label_predict(i)));
end
max_num = max(table(1,:));
result = zeros(3,max_num);
for j = 1:length(label)
    index = table(1,j);
    result(1,index) = index;
    result(2,index)=result(2,index)+1;
    result(3,index)=result(3,index)+table(2,j);
end
    result(3,:)=result(3,:)./result(2,:);
    for k = max_num:-1:1
        if(result(1,k)==0)
            result(:,k)=[];
        end
    end
    result = result';
    result = sortrows(result,3);
end