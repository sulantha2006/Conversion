ClassificationTable = readtable('Classification_Table.csv');
X=table2array(ClassificationTable(:,3:21));
Y=table2array(ClassificationTable(:,2));

X1 = X(:,1:10);
X2 = X(:,10:19);

t1 = templateTree('MinLeafSize',10);
rusTree1 = fitensemble(X1, Y, 'RUSBoost', 500, t1, 'LearnRate', 0.1, 'kFold', 10);
[Set1Values, Score1] = kfoldPredict(rusTree1);
t2 = templateTree('MinLeafSize',10);
rusTree2 = fitensemble(X2, Y, 'RUSBoost', 500, t1, 'LearnRate', 0.1, 'kFold', 10);
[Set2Values, Score2] = kfoldPredict(rusTree2);

PosteriorProb1 = double(bsxfun(@eq, Score1, max(Score1, [], 2)));
PosteriorProb2 = double(bsxfun(@eq, Score2, max(Score2, [], 2)));
YFit1 = PosteriorProb1(:,2);
YFit2 = PosteriorProb2(:,2);

tab = tabulate(Y);
bsxfun(@rdivide,confusionmat(Y,YFit1),tab(:,2))*100
bsxfun(@rdivide,confusionmat(Y,YFit2),tab(:,2))*100