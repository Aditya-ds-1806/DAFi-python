%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name :- MANAS KUMAR MISHRA
% Spectrogram analysis for DAFi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs are the Audio samples and sampling frequency
% Outputs are the Energy, frequencybin and Timebin
function [Energy , FrequencyBin, TimeBin] = OurSpectrogram(Audio, fsample)

    firstAudio = Audio(:,1);
    secondAudio = Audio(:,2);

    AverageAudio = (firstAudio+secondAudio)./2;
    
    %%%    specfication    %%%
    %1. Window type :- hanning
    %2. Window length :- 10240 samples
    %3. Overlapping :- 0 samples
    %4. Frequency range :- 0 to sampling frequency (Hz) (0,44100)--> [0, 2*pi]
    [s,FrequencyBin, TimeBin] = spectrogram(AverageAudio, hanning(10240),0,fsample);
    
    % s is a complex matrix.
    Energy = abs(s); %Energy is abs(s).
    
    figure;
    spectrogram(AverageAudio,hanning(10240),0, fsample,'yaxis');
    
end