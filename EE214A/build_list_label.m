%clear the workspace
clear;
%read the file list
fid = fopen('allFiles.txt');
data = textscan(fid,'%s');
fclose(fid);
x = data{1};
%initialize the counting index and arrays
index = 1;
%build the list and label for read-read lists.
for i = 1:350
    for j = i:350
        %store and read name of list
        left(index,1) = x(i);
        right(index,1) = x(j);
        %compare the value to decide whether it is the same person
        if(same_speaker(i,j,x))
            label(index,1) = 1;
        else
            label(index,1) = 0;
        end
        index = index + 1;
    end
end
[readtrainlist1,readtrainlist2,readtrainlabel,readtestlist1,readtestlist2,readtestlabel] = dividelist(left,right,label);
%repeat for phone-phone and read-phone
index = 1;
%build the list and label for phone-phone lists.
for i = 351:700
    for j = i:700
        %store and read name of list
        left(index,1) = x(i);
        right(index,1) = x(j);
        %compare the value to decide whether it is the same person
        if(same_speaker(i,j,x))
            label(index,1) = 1;
        else
            label(index,1) = 0;
        end
        index = index + 1;
    end
end
[phonetrainlist1,phonetrainlist2,phonetrainlabel,phonetestlist1,phonetestlist2,phonetestlabel] = dividelist(left,right,label);
index = 1;
%build the list and label for read-phone lists.
for i = 1:350 
    for j = 351:700
        %store and read name of list
        left(index,1) = x(i);
        right(index,1) = x(j);
        %compare the value to decide whether it is the same person
        if(same_speaker(i,j,x))
            label(index,1) = 1;
        else
            label(index,1) = 0;
        end
        index = index + 1;
    end
end
[mixtrainlist1,mixtrainlist2,mixtrainlabel,mixtestlist1,mixtestlist2,mixtestlabel] = dividelist(left,right,label);
%save lists and labels
save('lists_labels.mat');
function [trainlist1,trainlist2,trainlabel,testlist1,testlist2,testlabel] = dividelist(left,right,label)
%rearrange the list and label
randIndex = randperm(size(label,1));
left = left(randIndex,:);
right = right(randIndex,:);
label = label(randIndex,:);
len = length(label);
%divide the test and train list and label
trainlist1 = left(1:len-10000,:);
trainlist2 = right(1:len-10000,:);
trainlabel = label(1:len-10000,:);
testlist1 = left(len-9999:len,:);
testlist2 = right(len-9999:len,:);
testlabel = label(len-9999:len,:);
end
function [value] = same_speaker(i,j,x)
        stri = string(x(i));
        strj = string(x(j));
        %read string and get the value for the person
        i2 = sscanf(stri,('WavData/phone/%3d%1a_0_phone_%2d.wav'));
        j2 = sscanf(strj,('WavData/phone/%3d%1a_0_phone_%2d.wav'));
        value = (i2==j2);
end
