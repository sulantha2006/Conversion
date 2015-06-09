t = templateTree('MinLeafSize', 10);
ClassificationTable = readtable('Classification_Table.csv');
X=table2array(ClassificationTable(:,3:21));
Y=table2array(ClassificationTable(:,2));
X1 = X(:,[2,3,4,5,7,8,11,13,15,19]);
rus1 = fitensemble(X1, Y, 'RUSBoost', 250, t, 'LearnRate', 0.3, 'kFold', 30, 'CategoricalPredictors', [1]);
[value, score] = kfoldPredict(rus1);

pos1 = double(bsxfun(@eq, score, max(score, [], 2)));
yfit = pos1(:, 2);
tab = tabulate(Y);

val = bsxfun(@rdivide, confusionmat(Y, yfit), tab(:,2))*100
confusionMat = confusionmat(Y, yfit)
Fitness_F1 = 1 - getFitnessFromConfusionMat(confusionMat)

[X_p,Y_p,T_p,AUC,OPTROCPT,suby,subnames] = perfcurve(Y, score(:,2), '1');
AUC
plot(X_p,Y_p)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off