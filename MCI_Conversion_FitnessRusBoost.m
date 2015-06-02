function [ Fitness ] = MCI_Conversion_FitnessRusBoost( pop )

if sum(pop) < 1
    Fitness = 1;
    return
end

ClassificationTable = readtable('Classification_Table.csv');
X=table2array(ClassificationTable(:,3:21));
Y=table2array(ClassificationTable(:,2));

X1 = X(:, find(pop==1));
Y1 = Y;
ErrorMat = ones(1, 5); 
for idx = 1:length(ErrorMat)
    [model, Error] = GenerateRUSBoostModel(X1, Y1);
    ErrorMat(idx) = 1 - (Error(1,1) + 2*Error(2,2))/300;
end

Fitness = mean(ErrorMat);

end

