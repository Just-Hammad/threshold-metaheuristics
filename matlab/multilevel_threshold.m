function [thresholds, value] = multilevel_threshold(img, nClasses, criterion, pythonExe)
%MULTILEVEL_THRESHOLD  Exact multilevel Otsu/Kapur thresholds via threshmh.
%
%   [T, V] = MULTILEVEL_THRESHOLD(IMG, C) returns the C-1 globally optimal
%   Otsu thresholds for the uint8 image IMG, and the criterion value V.
%
%   [T, V] = MULTILEVEL_THRESHOLD(IMG, C, 'kapur') uses Kapur's entropy.
%
%   The result is a proven global optimum computed by dynamic programming,
%   not a heuristic search result. Runs in milliseconds for any practical C.
%
%   Tested against MATLAB R2015a, which has no Python interface; the bridge
%   is a subprocess plus two temporary CSV files.
%
%   Example:
%       img = imread('coins.png');
%       [t, v] = multilevel_threshold(img, 4);
%       seg = imquantize(img, t);

    if nargin < 3 || isempty(criterion), criterion = 'otsu'; end
    if nargin < 4 || isempty(pythonExe), pythonExe = 'python3'; end

    % Send the histogram rather than the image: it is 256 numbers instead of
    % megabytes, and it is all the objective depends on.
    counts = histc(double(img(:)), 0:255);

    histFile = [tempname '.csv'];
    outFile  = [tempname '.csv'];
    cleanup  = onCleanup(@() delete_if_present({histFile, outFile}));

    dlmwrite(histFile, counts(:)', 'delimiter', ',', 'precision', '%d');

    cmd = sprintf('%s -m threshmh.bridge --hist "%s" --classes %d --criterion %s --out "%s"', ...
                  pythonExe, histFile, nClasses, criterion, outFile);
    [status, output] = system(cmd);
    if status ~= 0
        error('multilevel_threshold:pythonFailed', ...
              'threshmh bridge failed (status %d):\n%s', status, output);
    end

    raw = dlmread(outFile);
    thresholds = raw(1:end-1)';
    value      = raw(end);
end

function delete_if_present(files)
    for k = 1:numel(files)
        if exist(files{k}, 'file'), delete(files{k}); end
    end
end
