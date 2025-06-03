A = imread('dog.jpeg');               % Read the image
X = double(rgb2gray(A));             % Convert RGB to gray, 256-bit to double
nx = size(X,1); ny = size(X,2);      % Get image dimensions

figure, subplot(2,2,1)
imagesc(X), axis off
colormap gray                        % Apply gray colormap
title('Original')
set(gcf, 'Position', [1400 100 1200 1600])

%% SVD Approximations
[U, S, V] = svd(X, 'econ');          % Compute SVD with economy mode

plotind = 2;
for r = [5 20 100]                   % Truncation values
    Xapprox = U(:,1:r)*S(1:r,1:r)*V(:,1:r)';  % Approximation using rank-r
    subplot(2,2,plotind), plotind = plotind + 1;
    imagesc(Xapprox), axis off, colormap gray
    title(['r=', num2str(r), ', ', num2str(100*r*(nx+ny)/(nx*ny),'%2.2f'), '% storage'])
end

%% Singular Values
figure, subplot(1,2,1)
semilogy(diag(S), 'k', 'LineWidth', 2), grid on
xlabel('r')
ylabel('Singular value, \sigma_r')
xlim([0 50])                         % Adjust x-axis limits
set(gca, 'FontSize', 14)

subplot(1,2,2)
plot(cumsum(diag(S))/sum(diag(S)), 'k', 'LineWidth', 2), grid on
xlabel('r')
ylabel('Cumulative energy captured')
set(gca, 'FontSize', 14)
