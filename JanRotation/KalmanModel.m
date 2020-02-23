classdef KalmanModel < handle
    properties
        A
        C
        muPrior
        initVar
        a
        totalI
        Ks
        Vars
        blankMus
    end
    
    methods
        function obj = KalmanModel(varargin)
            props = properties(obj);
            for v = varargin
                obj.(props{1}) = v{1};
                props = props(2:end);
            end
            
            if size(obj.initVar,2) ~= size(obj.initVar,1)
               obj.initVar = diag(obj.initVar);
            end
            obj.Vars = NaN([size(obj.initVar), obj.totalI]);
            obj.Ks = NaN([fliplr(size(obj.C)), obj.totalI]);
            obj.Vars(:,:, 1) = obj.initVar;
%           Get filter
            for i = 2:obj.totalI
                obj.Ks(:,:,i) = (obj.A*squeeze(obj.Vars(:,:,i-1))*obj.A'*obj.C')/(obj.C*obj.A*obj.Vars(:,:,i-1)*obj.A'*obj.C' + obj.C*obj.initVar*obj.C');
                obj.Vars(:,:,i) = (eye(size(obj.Ks(:,:,i)*obj.C,1)) - obj.Ks(:,:,i)*obj.C)*obj.A*obj.Vars(:,:,i-1)*obj.A';
            end
            
            obj.blankMus = NaN(length(obj.muPrior)+1, obj.totalI);
        end
        
        function Mus = runSim(obj, Zs, Ys)
%             LLike = @(Mu, Cov, Y) ((2*pi)^(-length(obj.muPrior)/2))*(det(Cov)^(-1/2))*exp(-((Y-Mu)'/inv(Cov)\(Y-Mu))/2);
            Mus = obj.blankMus;
            Mus(:, 1) = [obj.muPrior; 0];
            for i = 2:obj.totalI
                Mus(1:end-1,i) = obj.A*Mus(1:end-1,i-1) + obj.Ks(:,:,i)*(Ys(i) - obj.C*obj.A*Mus(1:end-1,i-1));
                Mus(end,i) = logsumexp([Mus(end,i-1), mvnpdf(Ys(:,i), obj.C*Mus(1:end-1,i), obj.C*obj.Vars(:,:,i)*obj.C')], 2);%LLike(obj.C*Mus(1:end-1,i), obj.C*obj.Vars(i)*obj.C', Ys(i))],2);
            end
            Mus(1:end-1,:) = Mus(1:end-1,:) - Zs;
            Mus(1:end-1,:) = Mus(1:end-1,:).^2;
            %bsxfun()
            return
        end
        
    end
    
    
end