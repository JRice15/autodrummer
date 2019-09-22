import sys
import numpy as np
import tensorflow as tf
import os

from os.path import dirname
global relativism_dir
relativism_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(relativism_dir)


from analysis import *
from recording_obj import Recording
from sampler import *
from input_processing import *
from output_and_prompting import *




class AutoDrummer(Analysis):

    def __init__(self, rec):
        super().__init__(rec)

        self.set_frame_fractions(1/20, 1/100)

        # get peaks from waveform
        frames = self.get_frames_mono()
        all_peaks = self.find_peaks(frames)
        self.peaks = self.filter_peaks(all_peaks)

        self.plot(peaks)

        # run modelling
        self.model()
    

    def read_model(self):
        pass


    def model(self):
        """
        run tensorflow model
        """
        # define optimizer behavior
        self.gradient_step = 0.01
        self.num_models = 10
        self.passes_per_model = 1000

        # convert constants to tensors
        self.rate = tf.constant(
            self.rate, 
            dtype=tf.float32, 
            name="sample rate")
        self.peaks = self.format_peaks(self.peaks)

        # define tf variables
        bpm = tf.Variable(tf.random.uniform(60, 120), dtype=tf.float32,  name='bpm')
        offset_factor = tf.Variable(tf.random.uniform(0, 1), dtype=tf.float32, name="offset factor")
        beat_weights = tf.Variable #TODO

        model = self.get_one_model(bpm, offset_factor)
        cost = self.model_cost(model, beat_weights)

        optimizer = tf.train.GradientDescentOptimizer(name="optimizer")
        optimization = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            file_writer = tf.summary.FileWriter(os.getcwd(), ts.graph)
            sess.run(init)
        
            best_model = []
            for _ in range(self.num_models):
                result = sess.run(optimization)


    def format_peaks(self):
        # make tensor
        formatted_peaks = tf.convert_to_tensor(
            self.peaks,
            dtype=tf.float32,
            name="peaks"
        )
        # sort by magnitude
        formatted_peaks = np_sort(
            formatted_peaks,
            column=1
        )
        # scale to 0-1 range
        formatted_peaks = scale(
            formatted_peaks,
            low=tf.reduce_min(
                formatted_peaks,
                axis=0
            )[1],
            high=tf.reduce_max(
                formatted_peaks,
                axis=0
            )[1]
        )
        # peaks as (sample_index, scaled slope), sorted by slope
        return format_peaks


    def get_one_model(self, bpm, offset_factor):
        """
        get one model using bpm and offset variables
        """
        offset = int(offset_factor * self.rate)
        samples_per_beat = 1 / (bpm * 60) * self.rate

        model_beats = []

        high_bound = max(self.peaks[:, 0]) + int(2 * self.rate)
        fractions = ['b', 'qb', 'hb', 'qb'] # aka quarternote, sixteenthnote, eighthnote, sixteenth
        eighths = [int(i * self.rate / 4) for i in range(4)]
        for i in range(-offset, high_bound, int(samples_per_beat)):
            for j in range(4):
                # each model beats as [index, beat type]
                model_beats.append([i + eighths[j], fractions[j]])

        return np.asarray(model_beats)


    def model_cost(self, model, beat_weights):
        """

        """
        # match peaks to model beats, starting with highest peaks
        peak_model_match = np.zeros(self.peaks.shape) # [ [peak, model_match] ]
        for i in range(self.peaks.shape[0]):
            diffs = (model[:,0] - self.peaks[i][0])
            min_ind = diffs.argmin()
            model_match = model[min_ind]
            model = np.delete(model, min_ind)
            peak_model_match[i] = np.asarray([self.peaks[i], model_match])





def scale(val, low, high, column=None):
    """
    return decimal percentage of distance val is from low to high.
    ie, 7 is 0.4 on scale from 5 to 10. if column given, scales np array
    """
    if column is None:
        return (val - low) / (high - low)
    else:
        val[:, column] = (val[:, column] - low) / (high - low)
        return val



class TrainAutodrummer():


    def __init__(self):
        pass

    def get_training_data(self):
        """
        read training_data.csv.
        returns [ [filename, bpm], ... ]
        """
        with open(dirname(__file__) + "/training_data.csv", "r") as f:
            data = [
                [j.strip() for j in i.split(",")][:2] for i in f.readlines()
                if i != "\n"
            ]
        return data








def super_sort(the_list, ind=None, ind2=None, high_to_low=False):
    """
    list, index1, index2, reverse.
    sorted by list[ind1][ind2] keys, for given indexs
    """
    if ind != None:
        if ind2 != None:
            return sorted(the_list, key=lambda i: i[ind][ind2], reverse=high_to_low)
        return sorted(the_list, key=lambda i: i[ind], reverse=high_to_low)
    return sorted(the_list, reverse=high_to_low)




def autodrummer_main():
    a = Recording(source=relativism_dir + '/sources/gtr_test_1.wav', name='test')
    AutoDrummer(a)









def autodrummer_help():
    print("\n*** AutoDrummer Help ***\n")
    print("Usage:\n")
    print("  Standard/Manual Entry: 'python3 autodrummer.py', then follow instructions\n")
    print("  Command Line Flags Entry: \n")
    print("    mode: m=[arg] / mode=[arg]")
    print("        args: 'f' - analyze file, 'r' - record\n")
    print("    playback: p= / playback=[arg]")
    print("        args: y/n - whether to play back file after recording/reading\n")
    print("    file: f= / filename=[arg], only used in file mode")
    print("        args: full name of sound file to analyze, which cannot contain spaces (only .wav files are accepted)\n")
    print("    recording duration: d= / duration=[arg], only used in record mode")
    print("        args: integer - recording time in seconds\n")
    print("    device: d= / device=[arg], used only in record mode")
    print("        args: integer - index of device to record from (run with no flags to get list of devices by index\n")
    print("Notes & Warnings:")
    print("  * AutoDrummer will not work well for selections with tempos below 60 bpm\n")
    sys.exit()


if __name__ == "__main__":
    if ("-h" in sys.argv) or ("-H" in sys.argv) or ("--help" in sys.argv) or ("help" in sys.argv):
        autodrummer_help()
    else:
        autodrummer_main()