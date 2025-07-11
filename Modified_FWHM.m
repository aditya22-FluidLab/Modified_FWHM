function [MScanData] = Modified_FWHM(filename,regs,modified,plt)
%________________________________________________________________________________________________________________________

% To run a file with filename: data.tif
% Enter in command window: [MScanData] = Modified_FWHM("data.tif")

% LIST OF MODIFICATIONS:

% [1.] REGISTRATION: Original algorithm used DFT registration by default which performs well
%  for low NSR images but fails for images with high NSR.
%  If user defines the input parameter "regs" as 1, then the Modified FWHM uses default DFT
%  registration else no registration is performed

% [2.] THRESHOLD: Original algorithm calculated FWHM based on an offset value
%   which was the minimum projected intensity. It works well for images
%   with low background noise but fails when contrast is low. Modified
%   algorithm defines offset as the 10th percentile of the projected
%   intensity. 
%   If user defines the input parameter "modified" as 1, then
%   the Modified FWHM uses the offset value based using the 10th percentile
%   approach.

% [3.] VISUAL OUTPUT: Original algorithm doesn't provide any visual output
%   to check is the estimated diameter results are accurate or not. The
%   modified version locates the vessel edge by identifying poitns which are at a 
%   distance of d/2 from the centerline (where d is the estimated
%   diameter). The located points are overlayed on the top of the vessel
%   and displayed as "yellow dots". If user defines the input parameter
%   "plt=1", then this algorithm displays the visual output.

%   Important: Required functions to obtain the visual output: imagei and superslider.

%   Additionally, the modified FWHM algorithm also generates the projected
%   image intensity profile.


% Original version of FWHM written/Edited by Kevin L. Turner
% The Pennsylvania State University, Dept. of Biomedical Engineering
% https://github.com/KL-Turner
%
% Adapted from code written by Dr. Patrick J. Drew: https://github.com/DrewLab
%________________________________________________________________________________________________________________________
%
%   Purpose: Analyzes the change in vessel diameter over time for a surface vessel (artery, vein, etc)
%________________________________________________________________________________________________________________________
%
%   Inputs: Click play - you will be prompted to select a single .tif stack. The stack used in this example is avaible
%           for download at https://drive.google.com/drive/folders/1YKzuRAuJtlcBx3zLxZods0PeUsiUrTlm?usp=sharing
%
%           The objective size for the provided stack is 16X, but is left as an input as this function is intended to be
%           shared and edited.
%
%           The vessel type for the provided stack is an artery (A).
%
%           You will then be prompted to draw two boxes, the first is an ROI around the diameter of the vessel. The second
%           is a line along the vessel center-axis. See README for more details.
%
%   Outputs: MScanData.mat struct, movie (5X speed), fig showing vessel diameter over time

tic
clc

[~,name,ext]=fileparts(filename);
imgfile=filename;

%% 
plt_default=0;
modified_default=0;
regs_default=0;

if nargin < 1
    error(['Usage: [MScandata] = ' mfilename ...
    '(imgfile,[regs],[modified],[plt])'])
end

if ~exist('regs','var') || isempty(regs)
    regs = regs_default;
end
if ~exist('modified','var') || isempty(modified)
    modified = modified_default;
end
if  ~exist('plt','var') || isempty(plt)
    plt = plt_default;
end


%% 
MScanData.notes.regs=regs; % Stores info if registration is to be performed or not
MScanData.notes.modified=modified; % Stores info if modified FWHM is to be used or not
MScanData.notes.filename=filename;
%% 
switch ext
    case '.tif'
        disp('You are analyzing the Tif file.')
        fileType = 1;
    case '.mat'
        disp('You are analyzing the 3D image array.')
        fileType = 0;
    otherwise
        disp('No valid selection made.')
        return;
end
MScanData.notes.fileType=fileType;
switch fileType
    case 1
        % Tifstack : Microscope
        if isequal(imgfile, 0)
            disp('No file selected.');
        else
            disp(['Loading: ' imgfile '...']);
            tifStack=imgfile;
            movieInfo=imfinfo(imgfile);
            numFrames=length(movieInfo);
            imgarray=zeros(movieInfo(1).Height, movieInfo(1).Width,numFrames, 'uint16');
            for k = 1:numFrames
                imgarray(:,:,k)=imread(imgfile,k);
            end
            MScanData.notes.imgarray=imgarray;
        end
    case 0
        % Microscope or Two photon data (.raw files) 
        disp(['Loading: ' imgfile '...']);
        loadedData=load(imgfile);
        fieldNames=fieldnames(loadedData);
        imgarray=(loadedData.(fieldNames{1})); % Use matlab functions like mat2gray in case of incorrect measurements 
        MScanData.notes.imgarray=uint16(imgarray);
end

if fileType==1

    imgarray=tifStack;
    MScanData.notes.header.frameWidth = num2str(movieInfo(1).Width);
    MScanData.notes.header.frameHeight = num2str(movieInfo(1).Height);
    MScanData.notes.header.numberOfFrames = length(movieInfo);
    MScanData.notes.xSize = str2double(MScanData.notes.header.frameWidth);
    MScanData.notes.ySize = str2double(MScanData.notes.header.frameHeight);

else

    MScanData.notes.header.frameWidth = size(imgarray,2);
    MScanData.notes.header.frameHeight = size(imgarray,1);
    numberOfFrames=size(imgarray,3);
    MScanData.notes.header.numberOfFrames = numberOfFrames;
    MScanData.notes.xSize = size(imgarray,2);
    MScanData.notes.ySize = size(imgarray,1);

end

magnification=1;
MScanData.notes.header.magnification =magnification;
rotation=0;
MScanData.notes.header.rotation = rotation;
frameRate=30;
MScanData.notes.header.frameRate=frameRate;

if fileType==1
    MScanData.notes.frameRate = 1/str2double(frameRate);
else
    MScanData.notes.frameRate = 1/(MScanData.notes.header.frameRate); %#ok<*ST2NM>
end

MScanData.notes.startframe = 1;
MScanData.notes.endframe = MScanData.notes.header.numberOfFrames;

%--------- Initializing to store threshold, offset and projection intensity data
MScanData.notes.extra=struct();


% Vessel type
MScanData.notes.vesselType='A';

% Initialize imageID field to prevent errors
MScanData.notes.imageID = 'Image';

micronsPerPixel=1;
MScanData.notes.micronsPerPixel = micronsPerPixel;

xFactor=1;
if fileType==1
    MScanData.notes.header.timePerLine = 1/(MScanData.notes.frameRate*str2num(MScanData.notes.header.frameHeight));
else
    MScanData.notes.header.timePerLine = 1/(MScanData.notes.frameRate*(MScanData.notes.header.frameHeight));
end

if fileType == 1
    if exist('tifStack', 'var') && ~isequal(tifStack, 0)
        image = imread(tifStack, 'TIFF', 'Index', 1);
    else
        disp('No valid TIF file selected.');
    end
elseif fileType == 0 
    image = imgarray(:,:,1);
end

% Draw vessel ROI and axis line
figure;
imagesc(double(image))
colormap("gray");
colorbar;
colormap('gray');
axis image
xlabel('pixels')
ylabel('pixels')

yString = 'y';
theInput = 'n';
xSize = size(image, 2);
ySize = size(image, 1);

area = impoly(gca, [1 1; 1 20; 20 20; 20 1]); %#ok<*IMPOLY>

setColor(area,'r');

while strcmp(yString, theInput) ~= 1
    drawnow()
    theInput = input('Is the diameter of the box ok? (y/n): ', 's'); disp(' ')
end

if strcmp(yString, theInput)
    get_API = iptgetapi(area);
    MScanData.notes.vesselROI.boxPosition.xy = get_API.getPosition();
    MScanData.notes.vesselROI.xSize = xSize;
    MScanData.notes.vesselROI.ySize = ySize;
    theInput = 'n';
end

diamAxis = imline(gca, round(xSize*[.25 .75]), round(ySize*[.25 .75])); %#ok<*IMLINE>
setColor(diamAxis,'g');

while strcmp(yString, theInput) ~= 1
    drawnow()
    theInput = input('Is the line along the diameter axis ok? (y/n): ', 's'); disp(' ')
end

if strcmp(yString, theInput)
    get_API = iptgetapi(diamAxis);
    MScanData.notes.vesselROI.vesselLine.position.xy = get_API.getPosition();
end
MScanData.notes.xFactor = xFactor;

if fileType==2 || fileType==0
    tifStack=imgarray;
end

disp('Analyzing vessel projections from defined polygons...'); disp(' ');
[MScanData] = GetDiameterFromMovie(MScanData, tifStack);

% Calc FWHM
try
    [MScanData] = FWHM_MovieProjection(MScanData, [MScanData.notes.startframe MScanData.notes.endframe]);
catch err
    disp(['FWHM calculation failed! Error: ' err.message]); disp(' ')
end

% Attempt to remove SMALL x-y motion artifacts, does not work well in vertical Z-plane
% 1 dural/vein, >40% changes spline, artery: >60% spline
% 2 dural/vein, >30% changes interpolate, artery: >50% interpolate
if strcmp(MScanData.notes.vesselType, 'D') || strcmp(MScanData.notes.vesselType, 'V')
    MScanData.data.vesselDiameter = RemoveMotion(MScanData.data.tempVesselDiameter, MScanData.notes.vesselROI.modalFixedDiameter, 2, 0.3);
else
    MScanData.data.vesselDiameter = RemoveMotion(MScanData.data.tempVesselDiameter, MScanData.notes.vesselROI.modalFixedDiameter, 2, 0.5);
end

% Figure of vessel diameter changes over time
figure;
t=MScanData.notes.startframe:MScanData.notes.endframe;
plot(t,MScanData.data.vesselDiameter, 'k')
title('Vessel diameter over time')
xlabel('Time (frames)')
ylabel('Diameter (Pixels)')
axis tight

disp('Diameter calculation for surface vessel example  - complete.'); disp(' ')

% Displaying visual output
Centerline=MScanData.notes.vesselROI.vesselLine.position.xy;
ROI=MScanData.notes.vesselROI.boxPosition.xy;
width=MScanData.data.vesselDiameter;

% Calculate centroid and slopes
centroid=[(Centerline(1,1)+Centerline(2,1))/2 (Centerline(1,2)+Centerline(2,2))/2];
slope_centerline=(Centerline(1,2)-Centerline(2,2))/(Centerline(1,1)-Centerline(2,1));
slope_perpline=-1/slope_centerline;

% Set up points
d=width/2;
num_frames=length(d);  % Number of frames/diameter values
x11=Centerline(1,1);
x22=Centerline(2,1);
y11=Centerline(1,2);
y22=Centerline(2,2);
P1=[x11,y11];
P2=[x22,y22];
xc=centroid(1);
yc=centroid(2);
Pc=[xc,yc];

% Number of points on both sides
n=5;
direcn=(P2-P1)/norm(P2-P1);
total_length=norm(P2-P1);
step_size=total_length/(4*n);

% Calculate points along centerline
points = zeros(2*n+1, 2);
points(1,:)=Pc;
for i = 1:n
    points(i+1,:)=Pc+(i*step_size*direcn);
    points(n+i+1,:)=Pc-(i*step_size*direcn);
end

% Calculate perpendicular points
theta=atan(slope_perpline);
num_points=size(points,1);
points_per_frame=num_points*2; % Points on both sides for each centerline point

all_points=zeros(points_per_frame,2,num_frames); % N×2×S array for imagei

x_ROI=ROI(:,1);
y_ROI=ROI(:,2);

for f=1:num_frames
    ij=1;
    for i=1:num_points

        % Compute the two side points
        x1=points(i,1)+d(f)*cos(theta);
        y1=points(i,2)+d(f)*sin(theta);

        x2=points(i,1)-d(f)*cos(theta);
        y2=points(i,2)-d(f)*sin(theta);

        % Points outside the ROI will be assigned as NaN
        if inpolygon(x1,y1,x_ROI,y_ROI)
            all_points(ij,1,f)=x1;
            all_points(ij,2,f)=y1;
        else
            all_points(ij,:,f)=NaN;
        end
        ij=ij+1;

        if inpolygon(x2,y2,x_ROI,y_ROI)
            all_points(ij,1,f)=x2;
            all_points(ij,2,f)=y2;
        else
            all_points(ij,:,f)=NaN;
        end
        ij=ij+1;
    end
end

% Storing points
MScanData.notes.extra.allpoints=all_points;

% Data visualization
if plt==1
    if fileType==1
        tifFile=MScanData.notes.filename;
        info=imfinfo(tifFile);
        numFrames=numel(info);
        for k=1:numFrames
            tifImage(:, :, k)=imread(tifFile,k);
        end
        hf=imagei({tifImage tifImage tifImage},tifImage,[], all_points);
    
    elseif fileType==0
        img=imgarray;
        hf=imagei({img img img},img,[],all_points);
    end

figure(hf);
hold on;

% Plot centerline
diamAxis=imline(gca,[Centerline(1,1) Centerline(2,1)],[Centerline(1,2) Centerline(2,2)]);
setColor(diamAxis,'g');

% Plot ROI
area = impoly(gca, [ROI(1,1) ROI(1,2);ROI(2,1) ROI(2,2);ROI(3,1) ROI(3,2);ROI(4,1) ROI(4,2)]);
setColor(area,'r');
hold off;

else
    disp('No animation generated (define plt=1 to generate).')
end
end


%% Opens the tiff file and gets the vessel projections from the defined polygons
function [MScanData] = GetDiameterFromMovie(MScanData, fileID)

fileType=MScanData.notes.fileType;

if fileType == 1
        MScanData.notes.firstFrame = imread(fileID, 'TIFF', 'Index', 1);
    
elseif fileType == 0 || fileType == 2
    imgarray = MScanData.notes.imgarray;
    MScanData.notes.firstFrame = imgarray(:,:,1);
end

fftFirstFrame = fft2(double(MScanData.notes.firstFrame));
X = repmat(1:MScanData.notes.xSize, MScanData.notes.ySize, 1);
Y = repmat((1:MScanData.notes.ySize)', 1, MScanData.notes.xSize);
MScanData.notes.vesselROI.projectionAngle = atand(diff(MScanData.notes.vesselROI.vesselLine.position.xy(:, 1))/diff(MScanData.notes.vesselROI.vesselLine.position.xy(:, 2)));
atand(diff(MScanData.notes.vesselROI.vesselLine.position.xy(:, 1))/diff(MScanData.notes.vesselROI.vesselLine.position.xy(:, 2)));

for theFrame = MScanData.notes.startframe:MScanData.notes.endframe

    if fileType == 1
        rawFrame = imread(fileID, 'TIFF', 'Index', theFrame);
    else
        rawFrame = imgarray(:,:,theFrame);
    end

    fftRawFrame = fft2(double(rawFrame));

    if MScanData.notes.regs==1

        [MScanData.notes.pixelShift(:, theFrame), ~] = DftRegistration(fftFirstFrame, fftRawFrame, 1);

        inpolyFrame = inpolygon(X + MScanData.notes.pixelShift(3, theFrame), Y + MScanData.notes.pixelShift(4, theFrame), MScanData.notes.vesselROI.boxPosition.xy(:, 1), MScanData.notes.vesselROI.boxPosition.xy(:, 2));
   

    elseif MScanData.notes.regs==0
        
        inpolyFrame = inpolygon(X , Y , MScanData.notes.vesselROI.boxPosition.xy(:, 1), MScanData.notes.vesselROI.boxPosition.xy(:, 2));

        
    end

    boundedrawFrame = rawFrame.*uint16(inpolyFrame);
    MScanData.notes.vesselROI.projection(theFrame, :) = radon(boundedrawFrame, MScanData.notes.vesselROI.projectionAngle);

    % Combine the projections
    combinedProjection(theFrame , :) = MScanData.notes.vesselROI.projection(theFrame, :); 
end
    
    % Combined projection figure
    figure;
    imagesc(combinedProjection); 
    colormap("gray");
    xlabel('Transverse distance (pixels)','FontSize',8);
    ylabel('Time (frames)','FontSize',8);
    colormap;  
    
end

%% Calculate diameter using FWHM and get the baseline diameter
function [MScanData] = FWHM_MovieProjection(MScanData, theFrames)

for f = min(theFrames):max(theFrames)
    % Add in a 5 pixel median filter
    filteredProjection = medfilt1(MScanData.notes.vesselROI.projection(f, :), 5);
    
    % Pass the frame index to CalcFWHM
    [MScanData, MScanData.data.rawVesselDiameter(f)] = CalcFWHM(filteredProjection, 1, [], MScanData, f);
end

MScanData.data.tempVesselDiameter = MScanData.data.rawVesselDiameter*MScanData.notes.xFactor;
[holdHist, d] = hist(MScanData.data.tempVesselDiameter, 0:.25:100);
[~, maxD] = max(holdHist);
MScanData.notes.vesselROI.modalFixedDiameter = d(maxD);

end

%% Calc full-width at half-max
function [MScanData,width] = CalcFWHM(data,smoothing,threshold,MScanData,frameIndex)

data = double(data(:));     % make sure this is column, and cast to double

% smooth data, if appropriate
if nargin < 2
    % smoothing not passed in, set to default (none)
    smoothing = 1;
end

if smoothing > 1
    data = conv2(data, rectwin(smoothing) ./ smoothing, 'valid');
end

% Calculate offset (min value)
if MScanData.notes.modified==1
    thr_baseline=prctile(nonzeros(data),10);
    offset=thr_baseline;
    threshold=max(data-offset)/2+offset;
else
offset = min(data);
end

% Calculate threshold if not provided
if nargin < 3 || isempty(threshold)
    threshold = max(data - offset) / 2 + offset;  % threshold is half max, taking offset into account
end

aboveI = find(data > threshold);    % all the indices where the data is above half max

if isempty(aboveI)
    % nothing was above threshold!
    width = 0;
    return
end

firstI = aboveI(1);                 % index of the first point above threshold
lastI = aboveI(end);                % index of the last point above threshold

if (firstI-1 < 1) || (lastI+1) > length(data)
    % interpolation would result in error, set width to zero and just return ...
    width = 0;
    return
end

% use linear interpolation to get a more accurate picture of where the max was
% find value difference between the point and the threshold value,
% and scale this by the difference between integer points ...
point1offset = (threshold-data(firstI-1)) / (data(firstI)-data(firstI-1));
point2offset = (threshold-data(lastI)) / (data(lastI+1)-data(lastI));

point1 = firstI-1 + point1offset;
point2 = lastI + point2offset;

width = point2-point1;

% Storing offset,threshold and intensity data
MScanData.notes.extra.threshold(frameIndex)=threshold;
MScanData.notes.extra.offset(frameIndex)=offset;
MScanData.notes.extra.data(frameIndex,:)=data;


end

%% Remove motion artifacts
function newVesselDiameter = RemoveMotion(vesselDiameter, baseline, diffThresh, rawThresh)
indx1 = find(diff(vesselDiameter) > diffThresh);
indx2 = find(abs((vesselDiameter - baseline)/baseline) > (rawThresh));
indx = union(indx1 + 1, indx2);   % indx: points need to be interpolated
indx0 = 1:length(vesselDiameter);
indx0(indx) = [];   % indx0: good points
count = 1;

if isempty(indx) ~= 1
    if indx0(1) ~= 1
        indx0 = [1:indx0(1) - 1, indx0];
    end
    
    for a = 1:length(indx0) - 1
        step = indx0(a + 1) - indx0(a);
        if step == 1
            newVesselDiameter(count) = vesselDiameter(count); %#ok<*AGROW>
        end
        
        if (step ~= 1)
            newVesselDiameter(count) = vesselDiameter(count);
            newVesselDiameter(count + 1:count + step - 1) = (vesselDiameter(indx0(a + 1)) + vesselDiameter(indx0(a)))/2;
        end
        
        count = count + step;
    end
    
    newVesselDiameter(count) = vesselDiameter(indx0(end));
    if indx(end) == length(vesselDiameter)
        newVesselDiameter(indx0(end) + 1:length(vesselDiameter)) = vesselDiameter(indx0(end));
    end
else
    newVesselDiameter = vesselDiameter;
end

end

%% Registers image via cross-correlation
function [output, Greg] = DftRegistration(buf1ft, buf2ft, usfac)
%________________________________________________________________________________________________________________________
% Utilized in analysis by Kevin L. Turner
% The Pennsylvania State University, Dept. of Biomedical Engineering
% https://github.com/KL-Turner
%
% Code unchanged with the exception of this title block for record keeping
%
%   Last Opened: February 23rd, 2019
%________________________________________________________________________________________________________________________
%
% function [output Greg] = dftregistration(buf1ft,buf2ft,usfac);
% Efficient subpixel image registration by crosscorrelation. This code
% gives the same precision as the FFT upsampled cross correlation in a
% small fraction of the computation time and with reduced memory
% requirements. It obtains an initial estimate of the crosscorrelation peak
% by an FFT and then refines the shift estimation by upsampling the DFT
% only in a small neighborhood of that estimate by means of a
% matrix-multiply DFT. With this procedure all the image points are used to
% compute the upsampled crosscorrelation.
% Manuel Guizar - Dec 13, 2007

% Portions of this code were taken from code written by Ann M. Kowalczyk
% and James R. Fienup.
% J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
% object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458
% (1990).

% Citation for this algorithm:
% Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
% "Efficient subpixel image registration algorithms," Opt. Lett. 33,
% 156-158 (2008).

% Inputs
% buf1ft    Fourier transform of reference image,
%           DC in (1,1)   [DO NOT FFTSHIFT]
% buf2ft    Fourier transform of image to register,
%           DC in (1,1) [DO NOT FFTSHIFT]
% usfac     Upsampling factor (integer). Images will be registered to
%           within 1/usfac of a pixel. For example usfac = 20 means the
%           images will be registered within 1/20 of a pixel. (default = 1)

% Outputs
% output =  [error,diffphase,net_row_shift,net_col_shift]
% error     Translation invariant normalized RMS error between f and g
% diffphase     Global phase difference between the two images (should be
%               zero if images are non-negative).
% net_row_shift net_col_shift   Pixel shifts between images
% Greg      (Optional) Fourier transform of registered version of buf2ft,
%           the global phase difference is compensated for.

% Default usfac to 1
if exist('usfac')~=1, usfac=1; end %#ok<*EXIST>

% Compute error for no pixel shift
if usfac == 0
    CCmax = sum(sum(buf1ft.*conj(buf2ft)));
    rfzero = sum(abs(buf1ft(:)).^2);
    rgzero = sum(abs(buf2ft(:)).^2);
    error = 1.0 - CCmax.*conj(CCmax)/(rgzero*rfzero);
    error = sqrt(abs(error));
    diffphase=atan2(imag(CCmax),real(CCmax));
    output=[error,diffphase];
    
    % Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
    % peak
elseif usfac == 1
    [m,n]=size(buf1ft);
    CC = ifft2(buf1ft.*conj(buf2ft));
    [max1,loc1] = max(CC);
    [~,loc2] = max(max1);
    rloc=loc1(loc2);
    cloc=loc2;
    CCmax=CC(rloc,cloc);
    rfzero = sum(abs(buf1ft(:)).^2)/(m*n);
    rgzero = sum(abs(buf2ft(:)).^2)/(m*n);
    error = 1.0 - CCmax.*conj(CCmax)/(rgzero(1,1)*rfzero(1,1));
    error = sqrt(abs(error));
    diffphase=atan2(imag(CCmax),real(CCmax));
    md2 = fix(m/2);
    nd2 = fix(n/2);
    if rloc > md2
        row_shift = rloc - m - 1;
    else
        row_shift = rloc - 1;
    end
    
    if cloc > nd2
        col_shift = cloc - n - 1;
    else
        col_shift = cloc - 1;
    end
    output=[error,diffphase,row_shift,col_shift];
    
    % Partial-pixel shift
else
    
    % First upsample by a factor of 2 to obtain initial estimate
    % Embed Fourier data in a 2x larger array
    [m,n]=size(buf1ft);
    mlarge=m*2;
    nlarge=n*2;
    CC=zeros(mlarge,nlarge);
    CC(m+1-fix(m/2):m+1+fix((m-1)/2),n+1-fix(n/2):n+1+fix((n-1)/2)) = ...
        fftshift(buf1ft).*conj(fftshift(buf2ft));
    
    % Compute crosscorrelation and locate the peak
    CC = ifft2(ifftshift(CC)); % Calculate cross-correlation
    [max1,loc1] = max(CC);
    [~,loc2] = max(max1);
    rloc=loc1(loc2);cloc=loc2;
    CCmax=CC(rloc,cloc);
    
    % Obtain shift in original pixel grid from the position of the
    % crosscorrelation peak
    [m,n] = size(CC); md2 = fix(m/2); nd2 = fix(n/2);
    if rloc > md2
        row_shift = rloc - m - 1;
    else
        row_shift = rloc - 1;
    end
    if cloc > nd2
        col_shift = cloc - n - 1;
    else
        col_shift = cloc - 1;
    end
    row_shift=row_shift/2;
    col_shift=col_shift/2;
    
    % If upsampling > 2, then refine estimate with matrix multiply DFT
    if usfac > 2
        %%% DFT computation %%%
        % Initial shift estimate in upsampled grid
        row_shift = round(row_shift*usfac)/usfac;
        col_shift = round(col_shift*usfac)/usfac;
        dftshift = fix(ceil(usfac*1.5)/2); %% Center of output array at dftshift+1
        % Matrix multiply DFT around the current shift estimate
        CC = conj(dftups(buf2ft.*conj(buf1ft),ceil(usfac*1.5),ceil(usfac*1.5),usfac,...
            dftshift-row_shift*usfac,dftshift-col_shift*usfac))/(md2*nd2*usfac^2);
        % Locate maximum and map back to original pixel grid
        [max1,loc1] = max(CC);
        [~,loc2] = max(max1);
        rloc = loc1(loc2); cloc = loc2;
        CCmax = CC(rloc,cloc);
        rg00 = dftups(buf1ft.*conj(buf1ft),1,1,usfac)/(md2*nd2*usfac^2);
        rf00 = dftups(buf2ft.*conj(buf2ft),1,1,usfac)/(md2*nd2*usfac^2);
        rloc = rloc - dftshift - 1;
        cloc = cloc - dftshift - 1;
        row_shift = row_shift + rloc/usfac;
        col_shift = col_shift + cloc/usfac;
        
        % If upsampling = 2, no additional pixel shift refinement
    else
        rg00 = sum(sum( buf1ft.*conj(buf1ft) ))/m/n;
        rf00 = sum(sum( buf2ft.*conj(buf2ft) ))/m/n;
    end
    error = 1.0 - CCmax.*conj(CCmax)/(rg00*rf00);
    error = sqrt(abs(error));
    diffphase=atan2(imag(CCmax),real(CCmax));
    % If its only one row or column the shift along that dimension has no
    % effect. We set to zero.
    if md2 == 1
        row_shift = 0;
    end
    if nd2 == 1
        col_shift = 0;
    end
    output=[error,diffphase,row_shift,col_shift];
end

% Compute registered version of buf2ft
if (nargout > 1)&&(usfac > 0)
    [nr,nc]=size(buf2ft);
    Nr = ifftshift((-fix(nr/2):ceil(nr/2)-1));
    Nc = ifftshift((-fix(nc/2):ceil(nc/2)-1));
    [Nc,Nr] = meshgrid(Nc,Nr);
    Greg = buf2ft.*exp(1i*2*pi*(-row_shift*Nr/nr-col_shift*Nc/nc));
    Greg = Greg*exp(1i*diffphase);
elseif (nargout > 1)&&(usfac == 0)
    Greg = buf2ft*exp(1i*diffphase);
end
return

    function out=dftups(in,nor,noc,usfac,roff,coff)
        % function out=dftups(in,nor,noc,usfac,roff,coff);
        % Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
        % a small region.
        % usfac         Upsampling factor (default usfac = 1)
        % [nor,noc]     Number of pixels in the output upsampled DFT, in
        %               units of upsampled pixels (default = size(in))
        % roff, coff    Row and column offsets, allow to shift the output array to
        %               a region of interest on the DFT (default = 0)
        % Recieves DC in upper left corner, image center must be in (1,1)
        % Manuel Guizar - Dec 13, 2007
        % Modified from dftus, by J.R. Fienup 7/31/06
        
        % This code is intended to provide the same result as if the following
        % operations were performed
        %   - Embed the array "in" in an array that is usfac times larger in each
        %     dimension. ifftshift to bring the center of the image to (1,1).
        %   - Take the FFT of the larger array
        %   - Extract an [nor, noc] region of the result. Starting with the
        %     [roff+1 coff+1] element.
        
        % It achieves this result by computing the DFT in the output array without
        % the need to zeropad. Much faster and memory efficient than the
        % zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
        
        [nr,nc]=size(in);
        % Set defaults
        if exist('roff')~=1, roff=0; end
        if exist('coff')~=1, coff=0; end
        if exist('usfac')~=1, usfac=1; end
        if exist('noc')~=1, noc=nc; end
        if exist('nor')~=1, nor=nr; end
        % Compute kernels and obtain DFT by matrix products
        kernc=exp((-1i*2*pi/(nc*usfac))*( ifftshift((0:nc-1)).' - floor(nc/2) )*( (0:noc-1) - coff ));
        kernr=exp((-i1*2*pi/(nr*usfac))*( (0:nor-1).' - roff )*( ifftshift((0:nr-1)) - floor(nr/2)  ));
        out=kernr*in*kernc;
        return
        
    end
toc
end
