function [ Fitness ] = MCI_Conversion_FitnessRusBoost_CSF( pop )

if sum(pop) < 1
    Fitness = 1;
    return
end
if ((length(pop)==8) && (sum(pop(4:8)) < 2))
    Fitness = 1;
    return
end


ClassificationTable = readtable('Classification_Table.csv');
X=table2array(ClassificationTable(:,3:10));
Y=table2array(ClassificationTable(:,2));

X1 = X(:, find(pop==1));
Y1 = Y;
[model, Error] = GenerateRUSBoostModel(X1, Y1);
Fitness = 1 - getFitnessFromConfusionMat(Error);

end