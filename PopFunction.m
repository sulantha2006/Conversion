function [pop] = PopFunction(GenomeLength,~,options)
    pop = (rand(options.PopulationSize, GenomeLength)> rand);
end

