function [ Fitness ] = MCI_Conversion_FitnessRusBoost( pop )

ClassificationTable = readtable('Classification_Table.csv');
X=table2array(ClassificationTable(:,3:21));
Y=table2array(ClassificationTable(:,2));

X1 = X(:, find(pop==1));
Y1 = Y;

[model, confMat] = GenerateRUSBoostModel(X1, Y1);
Fitness = 1-(confMat(1,1)+2*confMat(2,2))/300;

end

