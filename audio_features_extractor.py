'''
audio_feature_extractor.py
--------------------------
This script contains the AudioFeaturesExtractor class, which is used to extract features from audio files.
Usage:
    audio = AudioFeaturesExtractor(path)
    compute features:
        audio.mfcc()
        audio.f0()
        ...
    save features:
        audio.saveFeatures(features=['mfcc_', 'f0_', ...], basefolder='someFeaturesFolder')
        saves all the already computed features
        Note: Saving multiple features using saveFeatures() requires all features to have at least one common dimension axis for them to be concatenated as one array.
Authors:
    - Atharv Naik
    - Deepak Bajaj
'''


import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as transforms
import os
import csv
import pandas as pd
from tqdm import tqdm


class AudioFeaturesExtractor:
    '''
    This class represents an audio file and provides methods to extract various features from it, as well
    as a method to save selected features to csv file.\n
    Usage:
        audio = `AudioFeaturesExtractor(path)`\n
        Compute features:
            `audio.mfcc()`;
            `audio.f0()`;
            ...
        Save features:
            `audio.saveFeatures(features=['mfcc_', 'f0_', ...], basefolder='someFeaturesFolder')`\n
            Note: Saving multiple features using `saveFeatures()` requires all features to have at least one common dimension axis for them to be concatenated as one array.
    '''

    computable_features = [
        'mfcc_',
        'f0_',
        'chroma_',
        'spectral_centroid_',
        'spectral_bandwidth_',
        'spectral_rolloff_',
        'spectral_contrast_',
        'spectral_flatness_',
        'energy_',
        'zcr_',
        'spectral_flux_',
        'pause_rate_',
        'jitter_',
        'shimmer_',
        'f2_',
        'band_energy_ratio_',
    ]  # list of features that can be computed; add new features here as they are implemented; note: feature name must be feature_method_name + underscore

    def __init__(self, path: str, offset: float = 0, duration: float|None = None):
        self.path = path
        self.name = path.split('/')[-1].split('.')[0]
        self.data, self.sr = librosa.load(
            path, sr=None, mono=True, offset=offset, duration=duration)
        self.data = self.data / np.max(np.abs(self.data))
        self.duration = librosa.get_duration(y=self.data, sr=self.sr)

    def __getattr__(self, name):
        return self.__getattribute__(name)

    def mfcc(self, n_mfcc: int = 13) -> np.ndarray:
        """
        This function calculates the Mel-frequency cepstral coefficients (MFCC) of an audio signal using the
        librosa library in Python.

        :param n_mfcc: n_mfcc is the number of Mel Frequency Cepstral Coefficients (MFCCs) to compute. MFCCs
        are commonly used in audio signal processing and are a representation of the spectral envelope of a
        sound. The number of MFCCs to compute is a hyperparameter that can be, defaults to 13 (optional)

        :return: the computed Mel-frequency cepstral coefficients (MFCCs) of the audio data, with the number
        of coefficients specified by the `n_mfcc` parameter. The MFCCs are stored in a NumPy array and also
        assigned to the `self.mfcc` attribute of the object.
        """

        self.mfcc_ = librosa.feature.mfcc(
            y=self.data, sr=self.sr, n_mfcc=n_mfcc).T
        return self.mfcc_

    def f0(self, fmin: int = 50, fmax: int = 2000, frame_length: int = 2048, win_length: int = 1024, hop_length: int = 512) -> np.ndarray:
        """
        The function calculates the fundamental frequency (f0) of an audio signal using the Yin
        algorithm.

        :param fmin: The fmin parameter specifies the minimum frequency to be considered when calculating
        the fundamental frequency. The default value for fmin is 50 Hz, which is the lowest frequency
        audible to humans, defaults to 50 (optional)
        :param fmax: The fmax parameter specifies the maximum frequency to be considered when calculating
        the fundamental frequency. The default value for fmax is 2000 Hz, which is the highest frequency
        audible to humans, defaults to 2000 (optional)

        :return: The function `f0` returns the fundamental frequency (f0) of the audio data, which is
        calculated using the Yin algorithm. The f0 is stored in the `self.f0` attribute of the object and
        is also returned by the function.
        """

        self.f0_ = librosa.yin(self.data, fmin=fmin, fmax=fmax,
                               sr=self.sr, win_length=win_length, frame_length=frame_length, hop_length=hop_length).T.reshape(-1, 1)
        return self.f0_

    def chroma(self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512, n_chroma: int = 12) -> np.ndarray:
        """
        This function calculates the chroma feature of an audio signal using the chroma_stft function from
        the librosa library and returns the result.

        :return: The `chroma` feature of the audio data, which is computed using the `chroma_stft` function
        from the `librosa` library. The `chroma` feature represents the distribution of pitch classes in the
        audio signal.
        """

        self.chroma_ = librosa.feature.chroma_stft(
            y=self.data, sr=self.sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_chroma=n_chroma).T
        return self.chroma_

    def energy(self, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        This function calculates the energy of an audio signal using the root mean square (RMS) method and
        returns it as an array.

        :return: The `energy` method is returning the `self.energy` array, which is the root mean square
        (RMS) energy of the audio data stored in the `self.data` array.
        """

        self.energy_ = librosa.feature.rms(
            y=self.data, frame_length=frame_length, hop_length=hop_length).T
        return self.energy_

    def zcr(self, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        This function calculates the zero-crossing rate of an audio signal using the librosa library and
        returns the result.

        :return: The function `zcr` returns the zero-crossing rate feature of the audio data stored in the
        `self.data` variable. The feature is computed using the `librosa.feature.zero_crossing_rate`
        function and stored in the `self.zcr` variable. Finally, the function returns the `self.zcr`
        variable.
        """

        self.zcr_ = librosa.feature.zero_crossing_rate(
            y=self.data, frame_length=frame_length, hop_length=hop_length).T
        return self.zcr_

    def spectral_centroid(self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512) -> np.ndarray:
        """
        This function calculates and returns the spectral centroid of an audio signal using the librosa
        library.

        :return: the spectral centroid of the audio data as an array.
        """

        self.spectral_centroid_ = librosa.feature.spectral_centroid(
            y=self.data, sr=self.sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T
        return self.spectral_centroid_

    def spectral_bandwidth(self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512) -> np.ndarray:
        """
        This function calculates the spectral bandwidth of an audio signal using the librosa library and
        returns the result as an array.

        :return: the spectral bandwidth of the audio data as an array. The array is created using the
        librosa library's `spectral_bandwidth` function, which takes in the audio data and sample rate as
        inputs. The resulting array is then stored as an attribute of the instance using the `Array` class
        and returned.
        """

        self.spectral_bandwidth_ = librosa.feature.spectral_bandwidth(
            y=self.data, sr=self.sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T
        return self.spectral_bandwidth_

    def spectral_contrast(self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512, n_bands: int = 6, fmin: int = 200) -> np.ndarray:
        """
        This function calculates the spectral contrast of an audio signal using the librosa library and
        returns the result as an array.

        :param n_bands: The n_bands parameter specifies the number of frequency bands to be used when
        calculating the spectral contrast; defaults to 6 (optional)
        :param fmin: Frequency cutoff for the first bin `[0, fmin]` Subsequent bins will cover `[fmin, 2*fmin]`, `[2*fmin, 4*fmin]`, etc.

        :return: The function `spectral_contrast` is returning the spectral contrast feature of the audio
        data, which is calculated using the `librosa` library. The feature is stored in the
        `self.spectral_contrast` attribute of the object and is also returned by the function.
        """

        self.spectral_contrast_ = librosa.feature.spectral_contrast(
            y=self.data, sr=self.sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_bands=n_bands, fmin=fmin).T
        return self.spectral_contrast_

    def spectral_rolloff(self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512, roll_percent: float = 0.85) -> np.ndarray:
        """
        This function calculates the spectral rolloff of an audio signal using the librosa library and
        returns the result.

        :param roll_percent: The roll_percent parameter specifies the percentage of the total spectral
        energy that should lie below the rolloff frequency; defaults to 0.85 (85%) (optional)

        :return: the spectral rolloff of the audio data stored in the object. The spectral rolloff is a
        measure of the frequency below which a specified percentage of the total spectral energy lies.
        """

        self.spectral_rolloff_ = librosa.feature.spectral_rolloff(
            y=self.data, sr=self.sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, roll_percent=roll_percent).T
        return self.spectral_rolloff_

    def spectral_flux(self, lag: int = 1, max_size: int = 1) -> np.ndarray:
        """
        This function calculates the spectral flux of an audio signal using the librosa library.

        :param lag: The time lag for computing differences.
        :param max_size: size (in frequency bins) of the local max filter. Set to `1` to disable filtering

        :return: The `spectral_flux` array is being returned.
        """

        self.spectral_flux_ = librosa.onset.onset_strength(
            y=self.data, sr=self.sr, lag=lag, max_size=max_size).T.reshape(-1, 1)
        return self.spectral_flux_

    def spectral_flatness(self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512) -> np.ndarray:
        """
        This function calculates the spectral flatness of an audio signal using the librosa library and
        returns the result as an array.
        :return: the spectral flatness of the audio data as a NumPy array. The spectral flatness is a
        measure of how "flat" or "peaked" the power spectrum of the audio signal is, and is calculated as
        the geometric mean of the power spectrum divided by the arithmetic mean of the power spectrum.
        """

        self.spectral_flatness_ = librosa.feature.spectral_flatness(
            y=self.data, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T.reshape(-1, 1)
        return self.spectral_flatness_

    def pause_rate(self, threshold: float = 0.1, frame_length: int = 2048, hop_length: int = 512) -> float:
        """
        This function calculates the pause rate of an audio file based on a given energy threshold.

        :param threshold: The threshold is a value that is used to determine what level of energy in the
        audio is considered to be a pause. Any frames with energy levels below this threshold are considered
        to be silent frames and are counted towards the pause rate. The default value for the threshold is
        0.1, which means that any frames with energy levels below 10% of the maximum energy level are considered to be
        silent frames, defaults to 0.1 (optional)

        :return: the pause rate as a percentage, which is calculated by dividing the number of frames below
        the energy threshold by the total number of frames and multiplying by 100. The pause rate is stored
        in the `self.pause_rate` attribute of the object and is also returned by the function.
        """

        # Compute the energy of the audio
        energy = librosa.feature.rms(
            y=self.data, frame_length=frame_length, hop_length=hop_length)
        threshold = threshold * np.max(energy)

        # Calculate the number of frames below the threshold
        num_silent_frames = len(
            [frame for frame in energy[0] if frame < threshold])

        # Calculate the total number of frames
        total_frames = len(energy[0])

        # Calculate the pause rate as a percentage
        self.pause_rate_ = (num_silent_frames / total_frames) * 100

        return self.pause_rate_

    def jitter(self) -> float:
        """
        The function calculates the jitter value of a given data set.

        :return: the jitter value, which is calculated based on the differences between consecutive samples
        in the data array. The jitter value is a measure of the variation in the timing of the samples, and
        is calculated as the average absolute difference between consecutive differences, divided by the
        product of the length of the data array and the mean value of the data. The jitter value is stored
        as an attribute of the object and is also returned by the function.
        """

        # Calculate the differences between consecutive samples
        differences = np.diff(self.data)

        # Calculate the absolute differences between consecutive differences
        absolute_differences = np.abs(differences)

        # Calculate the average absolute difference
        average_difference = np.mean(absolute_differences)

        # Calculate the jitter value
        self.jitter_ = average_difference / \
            (len(self.data) * np.mean(self.data))

        return self.jitter_

    def shimmer(self) -> float:
        """
        The function calculates the shimmer value of a given data array by finding the average absolute
        difference between consecutive samples and dividing it by the mean of the data.

        :return: the shimmer value, which is calculated based on the average absolute difference between
        consecutive samples and the mean of the data. The shimmer value is stored in the `self.shimmer`
        attribute and is also returned by the function.
        """

        # Calculate the differences between consecutive samples
        differences = np.diff(self.data)

        # Calculate the absolute differences between consecutive samples
        absolute_differences = np.abs(differences)

        # Calculate the average absolute difference
        average_difference = np.mean(absolute_differences)

        # Calculate the shimmer value
        self.shimmer_ = average_difference / np.mean(self.data)

        return self.shimmer_

    def band_energy_ratio(self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512) -> float:
        """
        This function calculates the band energy ratio of an audio file using torchaudio and returns the
        result as an array.

        :return: The function `band_energy_ratio` returns the `self.band_energy_ratio` array, which is the
        ratio of the sum of spectrogram values in the frequency range from one-fourth to three-fourths of
        the total frequency bins, to the sum of all spectrogram values.
        """

        # using torchaudio
        waveform, sr = torchaudio.load(self.path)
        waveform = waveform.mean(dim=0)
        transform = transforms.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        spectrogram = transform(waveform)

        frequency_bins = spectrogram.shape[1]
        self.band_energy_ratio_ = spectrogram[:, frequency_bins //
                                              4:frequency_bins*3//4].sum(dim=1) / spectrogram.sum(dim=1)
        # mean ber value from tensor
        self.band_energy_ratio_ = self.band_energy_ratio_.mean(dim=0).item()
        return self.band_energy_ratio_

    def f2(self, n_fft: int = 1024, win_length: int = 1024, hop_length: int = 512, pre_max: int = 5, post_max: int = 5, pre_avg: int = 5, post_avg: int = 5, delta: float = 0.25, wait: int = 0) -> float:
        """
        This function calculates the second formant (F2) of an audio signal using the librosa library and
        returns the result as an array.

        A sample `n` is selected as an peak if the corresponding `x[n]` fulfills the following three conditions:

        1. `x[n] == max(x[n - pre_max:n + post_max])`
        2. `x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta`
        3. `n - previous_n > wait`

        where `previous_n` is the last sample picked as a peak (greedily).

        :param pre_max:  number of samples before `n` over which max is computed; defaults to 5 (optional)
        :param post_max: number of samples after `n` over which max is computed; defaults to 5 (optional)
        :param pre_avg: number of samples before `n` over which mean is computed; defaults to 5 (optional)
        :param post_avg: number of samples after `n` over which mean is computed; defaults to 5 (optional)
        :param delta: threshold offset for mean; defaults to 0.25 (optional)
        :param wait: number of samples to wait after picking a peak; defaults to 0 (optional)

        :return: The function `f2` returns the F2 peaks as an `Array` object.
        """

        # Compute the spectrogram
        spectrogram = np.abs(librosa.stft(
            self.data, n_fft=n_fft, hop_length=hop_length, win_length=win_length))

        # Identify peaks in the spectrogram
        peaks = []
        for i in range(spectrogram.shape[1]):
            column_peaks = librosa.util.peak_pick(
                x=spectrogram[:, i], pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait)
            peaks.extend(column_peaks)

        peaks = np.array(peaks)

        # Select the F2 peaks
        self.f2_ = (peaks[1:]).T  # Assuming the second peak corresponds to F2
        # get mean f2 across all frames
        self.f2_ = np.mean(self.f2_, axis=0)

        return self.f2_

    def computeAllFeatures(self) -> None:
        """
        This function computes all the features that have not yet been computed and stores them as
        attributes of the object.
        """

        for feature in tqdm(AudioFeaturesExtractor.computable_features):
            feature = feature[:-1]  # remove underscore
            self.__getattr__(feature)()

    def __check_features_computed(func: callable, *args, **kwargs) -> callable:
        """
        This function is a decorator that checks if the features specified in the `features` parameter have
        been computed before calling the function. If any of the features have not been computed, then a
        FeatureNotComputedError is raised.
        """

        def wrapper(self, features: list | str = [], *args, **kwargs):
            if type(features) == str:
                featuresList = [features]
            else:
                featuresList = features
            # check if any feature has not yet been computed; raise FeatureNotComputedError if so
            if any([not hasattr(self, feature) for feature in featuresList]):
                # get the names of the features that have not been computed
                missing_features = [
                    feature for feature in featuresList if not hasattr(self, feature)]
                raise FeatureNotComputedError(
                    f'Features {missing_features} accessed before being computed.')
            return func(self, features, *args, **kwargs)
        return wrapper

    @__check_features_computed
    def delta(self, feature: str, width: int = 9, order: int = 1, *args, **kwargs) -> np.ndarray:
        """
        This function applies the delta function on a given feature and returns the result.

        :param feature: The feature parameter specifies the name of the feature to which the delta function
        will be applied. The feature must be a 2D array, defaults to None (optional)
        :param width: The width parameter specifies the length of the filter window used to compute the
        delta feature. It is an optional parameter with a default value of 9. A larger value of width will
        result in a smoother delta feature, while a smaller value will capture more rapid changes in the
        original feature, defaults to 9 (optional)
        :param order: The order parameter in the Delta function specifies the order of the delta feature to
        be computed. Delta features are computed by taking the difference between adjacent frames of a
        feature, and higher order deltas are computed by taking the difference between adjacent frames of
        lower order deltas. The default value of order is 1, defaults to 1 (optional)

        :return: the delta feature array calculated using the librosa library. The delta feature array is
        wrapped in an Array object and is also being set as an attribute of the instance object. Finally,
        the delta feature array is being returned.
        """

        caller = feature + 'delta_'*order
        delta = librosa.feature.delta(
            self.__getattr__(feature), width=width, order=order)

        setattr(self, caller, delta)
        return delta

    @__check_features_computed
    def mean(self, feature: str) -> np.ndarray:
        """
        This function calculates the mean of a given feature and sets it as an attribute of an instance.
        :return: the mean of the input array as a numpy array with a shape of (1,1). It is also setting an
        attribute to the instance of the class with the name of the caller function + '_mean'.
        """

        mean = np.array([np.mean(self.__getattr__(feature))]).T

        setattr(self, feature + 'mean_', mean)
        return mean

    @__check_features_computed
    def var(self, feature: str) -> np.ndarray:
        """
        This function calculates the variance of a given feature and sets it as an attribute of an instance.
        :return: the variance of the input array as a numpy array with a shape of (1,1). It is also setting an
        attribute to the instance of the class with the name of the caller function + '_var'.
        """

        var = np.array([np.var(self.__getattr__(feature))]).T

        setattr(self, feature + 'var_', var)
        return var

    @__check_features_computed
    def std(self, feature: str) -> np.ndarray:
        """
        This function calculates the standard deviation of a given feature and sets it as an attribute of an instance.
        :return: the standard deviation of the input array as a numpy array with a shape of (1,1). It is also setting an
        attribute to the instance of the class with the name of the caller function + '_std'.
        """

        std = np.array([np.std(self.__getattr__(feature))]).T

        setattr(self, feature + 'std_', std)
        return std

    @__check_features_computed
    def getMeanFeaturesArray(self, features: list = []) -> np.ndarray:
        '''
        This function returns array of mean of the features specified in the array `features` as a ndarray.
        Note: The features must be computed first before calling this function.
        Sets the mean_features_array_ attribute of the object to the mean features array.

        :param features: A list of strings representing the names of the features to be returned

        :return: a NumPy array containing the mean of the features specified in the `features` parameter
        '''

        self.mean_features_array_ = np.mean(np.concatenate(
            [self.__getattr__(feature) for feature in features], axis=1), axis=0)
        return self.mean_features_array_

    @__check_features_computed
    def saveFeatures(self, features: list = [], basefolder: str = 'features') -> None:
        """
        This function saves features of an object as a CSV file in a specified folder. Note: The features
        must be 1D NumPy arrays.

        :param features: A list of strings representing the names of the features to be saved
        :param basefolder: The folder where the CSV file containing the features will be saved. If the
        folder does not exist, it will be created, defaults to 'features' (optional)
        """

        os.makedirs(basefolder, exist_ok=True)
        name = basefolder + "/" + self.name + '.csv'

        allfeatures = np.concatenate(
            [self.__getattr__(feature) for feature in features], axis=1)
        self.arrayfeatures = allfeatures

        with open(name, 'w', newline='') as file:
            writer = csv.writer(file)
            # if any feature has more than one column, then write column names before writing values
            if any([self.__getattr__(feature).shape[1] > 1 for feature in tqdm(features)]):
                # write column names
                firstrow = []
                for feature in features:
                    if self.__getattr__(feature).shape[1] > 1:
                        columns = [
                            feature + str(i+1) for i in range(self.__getattr__(feature).shape[1])]
                        firstrow.extend(columns)
                    else:
                        firstrow.append(feature)
                writer.writerow(firstrow)
            else:
                writer.writerow(features)
            writer.writerows(allfeatures)  # write values


class FeatureNotComputedError(AttributeError):
    """
    Feature accessed before being computed.
    """
    pass


if __name__ == '__main__':
    audio = AudioFeaturesExtractor('audio/105_AP1.wav')

    audio.computeAllFeatures()

    print(audio.delta('mfcc_', order=1))
    print(audio.mfcc_delta_)
    audio.getMeanFeaturesArray(features=['mfcc_', 'f0_'])
    print(audio.mean_features_array_)
    # audio.saveFeatures(features=['mfcc_', 'f0_'], basefolder='features')
    # audio.saveFeatures(features=audio.computable_features, basefolder='features')
