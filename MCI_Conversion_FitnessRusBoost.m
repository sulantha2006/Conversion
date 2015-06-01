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

Fitness_mat = ones(1, 5);
for idx = 1:length(Fitness_mat)
    [model, confMat] = GenerateRUSBoostModel(X1, Y1);
    Fitness_mat(idx) = 1-(confMat(1,1)+1.5*confMat(2,2))/250;
end
Fitness = mean(Fitness_mat);

end

