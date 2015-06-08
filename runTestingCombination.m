t = templateTree('MinLeafSize', 10);
ClassificationTable = readtable('Classification_Table.csv');
X=table2array(ClassificationTable(:,3:21));
Y=table2array(ClassificationTable(:,2));
X1 = X(:,[5,7,8,14,16,19]);
rus1 = fitensemble(X1, Y, 'RUSBoost', 250, t, 'LearnRate', 0.3, 'kFold', 10);
[value, score] = kfoldPredict(rus1);
pos1 = double(bsxfun(@eq, score, max(score, [], 2)));
yfit = pos1(:, 2);
tab = tabulate(Y);
val = bsxfun(@rdivide, confusionmat(Y, yfit), tab(:,2))*100
sum(Y)*val(2,1)/100
1-(val(1,1)+1.5*val(2,2))/250
[X_p,Y_p,T_p,AUC,OPTROCPT,suby,subnames] = perfcurve(Y, score(:,2), '1');
AUC
plot(X_p,Y_p)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off
T_p((X_p==OPTROCPT(1))&(Y_p==OPTROCPT(2)))