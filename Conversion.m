ClassificationTable = readtable('Classification_Table.csv');
global X;
global Y;
X=table2array(ClassificationTable(:,3:21));
Y=table2array(ClassificationTable(:,2));

NumberOfVars = 19;
gaOptions = gaoptimset('PopulationSize',50,...
                     'Generations',100,...
                     'PopulationType', 'bitstring',...
                     'MutationFcn',{@mutationuniform, 0.1},...
                     'CrossoverFcn', {@crossoverarithmetic,0.8},...
                     'EliteCount',2,...
                     'StallGenLimit',100,...
                     'PlotFcns',{@gaplotbestf},...  
                     'Display', 'iter'); 
                 
FitnessFunction = @MCI_Conversion_FitnessRusBoost;
chromosome=ga(FitnessFunction, NumberOfVars, gaOptions);
bestFeatures = find(chromosome==1);
