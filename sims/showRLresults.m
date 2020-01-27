contCount = cumsum(contains(trainStats.States(2:3:end), "cueA") + contains(trainStats.States(2:3:end), "cueB"));
momentCount = cumsum(contains(trainStats.States(2:3:end), "momentumChoice"));

% plot(trainStats.EpisodeIndex, contCount, 'LineWidth', 2, 'Color', 'r')
% hold on
% plot(trainStats.EpisodeIndex, momentCount, 'LineWidth', 2, 'Color', 'b')
% legend('Learning Choice', 'Momentum Choice')
plot(trainStats.EpisodeIndex, contCount-momentCount, 'LineWidth', 2, 'Color', 'g')
xlabel('Trial#')
ylabel('Preference (+ = learning, - = momentum)')
