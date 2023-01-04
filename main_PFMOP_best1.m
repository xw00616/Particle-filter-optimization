% 2018/8/18: bring the adaptive adjustment of the standard error of the
% mcmc proposal: std(j)=std_initial/j; (j is the index of iteration)
% Ref: Sequential Monte Carlo Simulated Annealing, Enlu Zhou etc.
clear;
clear all
close all


% f1='obj_fun_Kursawe_01';
% f2='obj_fun_Kursawe_02';
run_max=20;
Problems={'DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7','DTLZ8','DTLZ9'...
    'UF1','UF2','UF3','UF4','UF5','UF6','UF7',...
    'WFG1','WFG2','WFG3','WFG4','WFG5'};
%%---------------------------Initialization-------------------------------%%
for Prob =1
    
    Problem=Problems{Prob};
    
    for run=1:run_max
        % for run=1:10
        %%---------------------------Initialization-------------------------------%%
        fprintf('running... ... \r');
        
        %% Generate the weight vectors
        M=2 ;
        N=200;
        [W,SS] = UniformPoint(50,M);
        wei=unifrnd(0,1,size(W,1),1);
%                 wei=linspace(0.01,1,size(W,1));
        L=size(W,1);
        
        
        [Population.decs,Boundary,Coding] = P_objective1('init',Problem,M,N);
        dim=size(Boundary,2);
        pareto_set=ones(L,dim);
        pareto_front=ones(L,M);
        X.N=N;
        xmin=Boundary(2,:);
        xmax=Boundary(1,:);
        T = ceil(N/10);
        
        % Detect the neighbours of each solution
        % B = pdist2(W,W);
        % [~,B] = sort(B,2);
        % B = B(:,1:T);
        %% Generate random population
        normWN=1;
        Population.objs=P_objective1('value',Problem,M,Population.decs);
        % P_output2(Population.decs,1,'PF1',Problem,M,1);
        Zmax  = max(Population.objs,[],1);
        Z = min(Population.objs,[],1);
        g_old = max(abs(Population.objs-repmat(Z,N,1))./repmat(Zmax-Z,N,1).*W(1,:),[],2);
        
        tic;
        for jo=1:L  % index of outer loop
            jo
            fprintf('running in the %i th outer iteration \r',jo);
            if jo==1
                X.OldValues=Population.decs;
%                 X.TargetValues=exp(-g_old/wei(jo));
                                X.TargetValues=exp(-g_old);
                [GlobalOpt.value,ind]=max(X.TargetValues);
                GlobalOpt.X=X.OldValues(ind,:);
                X.Weight=X.TargetValues+1e-99;
            else
                % Tchebycheff approach with normalization
                Offspring.decs=X.NewValues;
                Offspring.objs=P_objective1('value',Problem,M,Offspring.decs);
                
                Zmax  = max([Population.objs;Offspring.objs],[],1);
                Z = min(Z,min(Offspring.objs));
                
                g_old = max(abs(Offspring.objs-repmat(Z,N,1))./repmat(Zmax-Z,N,1).*W(jo-1,:),[],2);
                g_new = max(abs(Offspring.objs-repmat(Z,N,1))./repmat(Zmax-Z,N,1).*W(jo,:),[],2);
                
%                 X.TargetValues=exp(-g_new/wei(jo));
                %                 X.TargetValues=exp(-g_new);
                [GlobalOpt.value,ind]=max(X.TargetValues);
                GlobalOpt.X=X.NewValues(ind,:);
                
%                 X.ProposalValues=exp(-g_old/wei(jo-1))+1e-99;
                                X.ProposalValues=exp(-g_old)+1e-99;
                X.Weight=X.TargetValues./X.ProposalValues;
            end
            
            %normilizing
            X.NormalizedWeight=X.Weight/sum(X.Weight);
            %effective sample size
            sum(X.NormalizedWeight.^2)
            ESS=1/sum(X.NormalizedWeight.^2);
            Th=N/2;
            if ESS<Th
                % resampling
                outIndex = residualR(1:X.N, X.NormalizedWeight);
                X.NewValues =X.OldValues(outIndex,:);
                %%%%Metropolis sampling%%%%%%%%%%%%%%
            else
                X.NewValues=X.OldValues;
            end
            outIndex = residualR(1:X.N, X.NormalizedWeight); % resampling
            X.NewValues =X.OldValues(outIndex,:);
            %%%%Metropolis sampling%%%%%%%%%%%%%%
            for i=1:X.N
                X_old=X.NewValues(i,:);
                MatingPool=[GlobalOpt.X;mean(X.NewValues)];
                
                X_new = P_generator(MatingPool,Boundary,'Real',1);
                FunctionValue_old=P_objective1('value',Problem,M,X_old);
                FunctionValue_new=P_objective1('value',Problem,M,X_new);
                Zmax  = max(Zmax,FunctionValue_new);
                Z = min(Z,FunctionValue_new);
                g_1 = max(abs(FunctionValue_old-Z)./(Zmax-Z).*W(jo,:),[],2);
                g_2 = max(abs(FunctionValue_new-Z)./(Zmax-Z).*W(jo,:),[],2);
                
%                 X_old_Target=exp(-g_1/wei(jo));
%                 X_new_Target=exp(-g_2/wei(jo));
                                X_old_Target=exp(-g_1);
                                X_new_Target=exp(-g_2);
                
                
                if X_new_Target>GlobalOpt.value
                    GlobalOpt.value=X_new_Target;
                    GlobalOpt.X=X_new;
                end
                
                if rand<X_new_Target/X_old_Target
                    X.NewValues(i,:) = X_new;
                    X.Target(i)=X_new_Target;
                else
                    X.Target(i)=X_old_Target;
                end
            end
            %     end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            Population.decs=[Population.decs;X.NewValues];
            Population.objs=[Population.objs;P_objective1('value',Problem,M,X.NewValues)];
            P_output2(Population.decs,1,'PF2',Problem,M,run)
            run
            Problem
        end
        
    end
    
end
% time_elapsed2=toc;
% figure,
% %plot(pareto_front(:,1),pareto_front(:,2),'o',pf1,pf2,'-');legend('PFOPS','NSGA-II');grid on;
% P_output2(Population.decs,time_elapsed2,'PF1',Problem,M,2);

% figure,
% plot(ESS);grid on;
%save Res_pfops N L pareto_front time_elapsed2;
