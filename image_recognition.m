%clear workspace, close all figures, and clear command window
clear all;
close all;
clc;

%initialize webcam
cam = webcam;

%create a figure for real-time gesture recognition
figure('Name', 'Real-Time Gesture Recognition', 'NumberTitle', 'off', ...
       'Position', [100, 100, 800, 600]);

%initialize variables for tracking and gesture recognition
prevCentroid = [];
gesture = '';
trackingActive = false;

while true
    %capture a snapshot from the webcam
    inputImage = snapshot(cam);

    %display the original RGB image
    subplot(2, 2, 1);
    imshow(inputImage);
    title('Original RGB Image');

    %extract RGB channels
    redChannel = inputImage(:, :, 1);
    greenChannel = inputImage(:, :, 2);
    blueChannel = inputImage(:, :, 3);

    %define color threshold ranges
    redThreshMin = 0;
    redThreshMax = 100;
    greenThreshMin = 0;
    greenThreshMax = 100;
    blueThreshMin = 120;
    blueThreshMax = 255;

    %create binary masks based on the thresholds
    redMask = (redChannel >= redThreshMin) & (redChannel <= redThreshMax);
    greenMask = (greenChannel >= greenThreshMin) & (greenChannel <= greenThreshMax);
    blueMask = (blueChannel >= blueThreshMin) & (blueChannel <= blueThreshMax);
    binaryMask = redMask & greenMask & blueMask;
    binaryMask = uint8(binaryMask);

    %fill holes in the binary mask
    binaryMask = imfill(binaryMask, 'holes');
    binaryImage = imbinarize(binaryMask);

    %display the thresholded binary image
    subplot(2, 2, 2);
    imshow(binaryImage);
    title('Thresholded Image');

    %perform morphological closing to clean the binary image
    structuringElement = strel('disk', 20);
    cleanedBinaryImage = imclose(binaryImage, structuringElement);

    %initialize boundary image
    boundaryImage = false(size(cleanedBinaryImage));

    %compute region properties
    properties = regionprops(cleanedBinaryImage, 'Centroid', 'PixelList');

    %process centroid and gesture detection
    if ~isempty(properties)
        %extract the centroid and pixel list of the largest region
        centroid = properties(1).Centroid;
        pixelList = properties(1).PixelList;

        %calculate the medoid from the pixel list
        distances = pdist2(pixelList, pixelList);
        sumDistances = sum(distances, 2);
        [~, medoidIndex] = min(sumDistances);
        medoid = pixelList(medoidIndex, :);

        %display cleaned binary image with annotations
        subplot(2, 2, 3);
        imshow(cleanedBinaryImage);
        hold on;
        plot(centroid(1), centroid(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        plot(medoid(1), medoid(2), 'bo', 'MarkerSize', 10, 'LineWidth', 2);

        %generate and overlay boundaries
        boundaryStructElement = strel('disk', 5);
        gradientImage = imgradient(cleanedBinaryImage);
        boundaryImage = imclose(gradientImage, boundaryStructElement) > 0;
        [boundaries, ~] = bwboundaries(boundaryImage, 'noholes');
        for k = 1:length(boundaries)
            boundary = boundaries{k};
            plot(boundary(:, 2), boundary(:, 1), 'g', 'LineWidth', 1);
        end
        hold off;
        title('Cleaned Thresholded Image with Annotations');

        %detect gesture based on centroid movement
        if ~isempty(prevCentroid)
            delta = centroid - prevCentroid;
            if abs(delta(1)) > abs(delta(2))
                if delta(1) > 20
                    gesture = 'Right Swipe';
                elseif delta(1) < -20
                    gesture = 'Left Swipe';
                end
            elseif abs(delta(2)) > abs(delta(1))
                if delta(2) > 20
                    gesture = 'Down Swipe';
                elseif delta(2) < -20
                    gesture = 'Up Swipe';
                end
            end
        end
        prevCentroid = centroid;
        trackingActive = true;
    else
        %reset tracking if no object is detected
        trackingActive = false;
        prevCentroid = [];
        gesture = '';
    end

    %apply binary mask to original image and overlay boundary
    binaryMaskRgb = uint8(cat(3, cleanedBinaryImage, cleanedBinaryImage, cleanedBinaryImage));
    resultImage = inputImage .* binaryMaskRgb;
    resultImage(repmat(boundaryImage, [1, 1, 3])) = 255;

    %display the annotated original image with gesture information
    subplot(2, 2, 4);
    imshow(resultImage);
    hold on;

    if ~isempty(properties)
        plot(centroid(1), centroid(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        plot(medoid(1), medoid(2), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
    end

    hold off;
    title(['Annotated Image - Gesture: ', gesture]);

    %update the figure
    drawnow;
end

%release the webcam
clear cam;
