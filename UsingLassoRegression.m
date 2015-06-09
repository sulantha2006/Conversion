ClassificationTable = readtable('Classification_Table.csv');
X=table2array(ClassificationTable(:,3:21));
Y=table2array(ClassificationTable(:,2));

priors = ones(size(Y, 1), 1);
priors(find(Y==0)) = 1/sum(Y==0);
priors(find(Y==1)) = 1/sum(Y==1);

[B FitInfo] = lassoglm(X, Y, 'binomial', 'CV', 10, 'Weights', priors);
minpts = find(B(:,FitInfo.IndexMinDeviance))
sd1pts = find(B(:,FitInfo.Index1SE))

lassoPlot(B,FitInfo,'plottype','CV');