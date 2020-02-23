function plotMSE(MusLL, dims, Vars, allT, labels)
    [MSE, mVars] = getMSE(MusLL, dims, Vars);
    width = 2;
    
    colors = linspecer(size(MSE,1)*size(MSE,2));
    
    for SW = 1:size(MSE,1)
        for KM = 1:size(MSE,2)
            subplot(2, 2, 1)
            plot(allT, squeeze(MSE(SW, KM, 1, :)), 'Color', colors((SW-1)*size(MSE,2) + KM, :), 'LineWidth', width, 'DisplayName', sprintf('World_{%s}: Model_{%s}', labels{1, SW}, labels{2, KM}))
            hold on
            if SW == size(MSE, 1)
                plot(allT, squeeze(mVars(KM, 1,:)), 'Color', colors((SW-1)*size(MSE,2) + KM, :), 'LineWidth', width, 'LineStyle', ':', 'DisplayName', sprintf('Predicted: Model_{%s}', labels{2, KM}));
            end
            xlabel("time")
            ylabel("mean square error")
            title("Position MSE")

            if size(MSE,3) > 3
                subplot(2, 2, 2)
                plot(allT, squeeze(MSE(SW, KM, 2, :)), 'Color', colors((SW-1)*size(MSE,2) + KM, :), 'LineWidth', width, 'DisplayName', sprintf('World_{%s}: Model_{%s}', labels{1, SW}, labels{2, KM}))
                hold on
                if SW == size(MSE, 1)
                    plot(allT, squeeze(mVars(KM, 2,:)), 'Color', colors((SW-1)*size(MSE,2) + KM, :), 'LineWidth', width, 'LineStyle', ':', 'DisplayName', sprintf('Predicted: Model_{%s}', labels{2, KM}));
                end
                xlabel("time")
                ylabel("mean square error")
                title("Object Distance MSE")
            end

            subplot(2, 2, 3)
            plot(allT, squeeze(MSE(SW, KM, end-1, :)), 'Color', colors((SW-1)*size(MSE,2) + KM, :), 'LineWidth', width, 'DisplayName', sprintf('World_{%s}: Model_{%s}', labels{1, SW}, labels{2, KM}))
            hold on
            if SW == size(MSE, 1)
                plot(allT, squeeze(mVars(KM, end-1,:)), 'Color', colors((SW-1)*size(MSE,2) + KM, :), 'LineWidth', width, 'LineStyle', ':', 'DisplayName', sprintf('Predicted: Model_{%s}', labels{2, KM}));
            end
            xlabel("time")
            ylabel("mean square error")
            title("Velocity MSE")

            subplot(2, 2, 4)
            plot(allT, squeeze(MSE(SW, KM, end, :)), 'Color', colors((SW-1)*size(MSE,2) + KM, :), 'LineWidth', width, 'DisplayName', sprintf('World_{%s}: Model_{%s}', labels{1, SW}, labels{2, KM}))
            hold on
            if SW == size(MSE, 1)
                plot(allT, squeeze(mVars(KM, end,:)), 'Color', colors((SW-1)*size(MSE,2) + KM, :), 'LineWidth', width, 'LineStyle', ':', 'DisplayName', sprintf('Predicted: Model_{%s}', labels{2, KM}));
            end
            xlabel("time")
            ylabel("Log Likelihood")
            title("Overall Model Performance")
        end
    end
    legend
    return
end

function [MSE, MVar] = getMSE(MusLL, dims, Vars)  
    MVar = NaN(size(Vars,4), size(Vars,1)/dims, size(Vars,3));
    for i = 1:size(Vars,4)
        MVar(i, :, :) = nansum(reshape(cell2mat(arrayfun(@(j) diag(Vars(:,:,j,i)), 1:size(Vars,3), 'UniformOutput', false)), dims, [], size(Vars,3)), 1);
    end
    MSE = cat(3, squeeze(nansum(reshape(MusLL(:,:,1:end-1,:), 2, 2, dims, [], size(MusLL,4)), 3)), MusLL(:,:,end,:));
    
    return
end