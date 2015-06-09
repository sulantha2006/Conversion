clear all;clc;

ClassificationTable = readtable('Classification_Table.csv');
%%
Age = table2array(ClassificationTable(:,3));
Gender = nominal(table2array(ClassificationTable(:,4)));
Gender_dv = dummyvar(Gender);
X_temp=table2array(ClassificationTable(:,3:21));
X = [Age Gender_dv X_temp];
Y=table2array(ClassificationTable(:,2));

priors = ones(size(Y, 1), 1);
priors(find(Y==0)) = 1/sum(Y==0);
priors(find(Y==1)) = 1/sum(Y==1);

[B FitInfo] = lassoglm(X, Y, 'binomial', ...
    'CV', 20,...
    'Weights', priors, ...
    'NumLambda', 100);

minpts = find(B(:,FitInfo.IndexMinDeviance))
sd1pts = find(B(:,FitInfo.Index1SE))

lassoPlot(B,FitInfo,'plottype','CV');
%%
TableVarNames = ClassificationTable.Properties.VariableNames;
name_minpts = minpts + 2;
X1 = X(:,minpts');
X1VarNames = TableVarNames(name_minpts');
%%
t = templateTree('MinLeafSize', 15);
if ~any(strcmp(X1VarNames, 'PTGENDER'))
    rus1 = fitensemble(X1, Y, 'RUSBoost', 250, t, ...
    'LearnRate', 0.3, 'kFold', 10, 'PredictorNames', X1VarNames);
else
    rus1 = fitensemble(X1, Y, 'RUSBoost', 250, t, ...
    'LearnRate', 0.3, 'kFold', 10, 'PredictorNames', X1VarNames, ...
    'CategoricalPredictors', {'PTGENDER'});
end
[value, score] = kfoldPredict(rus1);

pos1 = double(bsxfun(@eq, score, max(score, [], 2)));
yfit = pos1(:, 2);
tab = tabulate(Y);
val = bsxfun(@rdivide, confusionmat(Y, yfit), tab(:,2))*100
confusionMat = confusionmat(Y, yfit)
Fitness_F1 = 1 - getFitnessFromConfusionMat(confusionMat)

[X_p,Y_p,T_p,AUC,OPTROCPT,suby,subnames] = perfcurve(Y, score(:,2), '1',...
    'Prior', 'empirical', 'Cost', [0,0.7;0.35,0]);
AUC
figure;
plot(X_p,Y_p)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off
