import sys
import numpy as np
import tensorflow as tf
import os
import random as rd

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
        # self.plot(all_peaks, plot_type="scatter", title="All Peaks")

        self.peaks = self.filter_peaks(all_peaks)
        # self.plot(self.peaks, plot_type="scatter", title="Filtered Peaks")

        # run modelling
        self.run()


    def format_peaks(self, peaks):
        # scale to 0-1 range
        high = np.percentile(peaks[:,1], 80)
        peaks[:,1] = peaks[:,1] / high
        peaks[:,1][peaks[:,1] > 1] = 1
        # self.plot(peaks, plot_type="scatter", title="Formatted Peaks")

        # make tensor
        formatted_peaks = tf.convert_to_tensor(
            peaks,
            dtype=tf.float32,
            name="tensor_peaks"
        )

        # peaks as (sample_index, scaled slope), sorted by slope
        return formatted_peaks


    def run(self):
        """
        run tensorflow model
        """
        # define optimizer behavior
        self.gradient_step = 0.01
        self.num_models = 100
        self.passes_per_model = 50

        # tf constants
        self.tf_peaks = self.format_peaks(self.peaks)
        self.tf_rate = tf.constant(self.rate, dtype=tf.float32, name="sample_rate")
        self.tf_pi = tf.constant(np.pi, dtype=tf.float32, name="pi")
        beat_factors = tf.constant( 
            # [beat, hb, qb]
            [1.0, 1/2, 1/4],
            dtype=tf.float32,
            name="beat_factors"
        )

        # tf variables
        bpm = tf.Variable(tf.random.uniform([1], 60, 180), dtype=tf.float32,  name='bpm')
        offset_factor = tf.Variable(tf.random.uniform([1], 0, 1), dtype=tf.float32, name="offset_factor")
        beat_wave_weights = tf.Variable(
            [
                # [beat, halfbeat, quarterbeat]
                tf.random.uniform([1], 0.3, 0.7),
                tf.random.uniform([1], 0.2, 0.6),
                tf.random.uniform([1], 0.1, 0.5),
            ],
            dtype=tf.float32,
            name="beat_weights"
        )

        # preparing variables
        samples_per_beat = tf.math.round((1 / (bpm * 60) * self.rate), name="samples_per_beat")
        beat_lengths = tf.math.round(beat_factors * samples_per_beat, name="beat_lengths")
        offset_length = tf.math.round(samples_per_beat * offset_factor, name="offset_length")

        # calculating cost
        cost = self.model_cost(beat_lengths, offset_length, beat_wave_weights)

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(
            self.gradient_step,
            name="optimizer"
        )
        gradients = optimizer.compute_gradients(cost)
        optimization = optimizer.minimize(cost)

        # running
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            file_writer = tf.summary.FileWriter(dirname(os.path.abspath(__file__)) + "/tboard_logs", sess.graph)
        
            best_models = []
            
            for m in range(self.num_models):
                # reset variables
                sess.run(init)
                print("model {0}".format(m))
                for p in range(self.passes_per_model):
                    sess.run(optimization)
                    weights = sess.run(beat_wave_weights)
                    print("  pass {0}, cost {1:.5f}, bpm {2:.5f}, offset_factor {3:.5f}, weights: {4:.5f} {5:.5f} {6:.5f}".format(
                        p, sess.run(cost), sess.run(bpm)[0], sess.run(offset_factor)[0], weights[0][0], weights[1][0], weights[2][0]))
                best_models.append({
                    "cost": sess.run(cost),
                    "bpm": sess.run(bpm)[0],
                    "offset": sess.run(offset_factor)[0],
                    "weights": sess.run(beat_wave_weights)
                })
        print("")
        for i in best_models:
            print("cost", i['cost'])
            print("    bpm", i['bpm'])
            print("    offset", i['offset'])
            print("    weights", i['weights'][0][0], i['weights'][1][0], i['weights'][2][0])


    # good sigmoid: 1 / 1 + 100exp(-10x)


    def model_cost(self, beat_lengths, offset_length, beat_wave_weights):
        wave_beat = self.wave_distribution_model(
            indexes=self.tf_peaks[:,0],
            wavelength=beat_lengths[0],
            peak_index=offset_length,
            wave_weight=beat_wave_weights[0],
            name="whole_beat_wave"
        )
        wave_halfbeat = self.wave_distribution_model(
            indexes=self.tf_peaks[:,0],
            wavelength=beat_lengths[1],
            peak_index=offset_length,
            wave_weight=beat_wave_weights[1],
            name="half_beat_wave"
        )
        wave_quarterbeat = self.wave_distribution_model(
            indexes=self.tf_peaks[:,0],
            wavelength=beat_lengths[2],
            peak_index=offset_length,
            wave_weight=beat_wave_weights[2],
            name="qrtr_beat_wave"
        )
        full_cost_wave = wave_beat + wave_halfbeat + wave_quarterbeat
        return tf.reduce_sum(
            tf.abs(
                self.tf_peaks[:,1] - full_cost_wave
            ),
            name="cost"
        )


    def wave_distribution_model(self, indexes, wavelength, peak_index, wave_weight, name):
        """
        all values in samples, except amp as factor.
        oscillates between 0 and around 1 (depends on wave weights variables).
        see https://www.desmos.com/calculator/ygx0cdgm7p for example of this func
        """
        return tf.math.multiply(
            wave_weight,
            tf.math.pow(
                tf.math.cos( 
                    (indexes - peak_index) * 2 * self.tf_pi / wavelength
                ),
                16
            ),
            name=name
        )
        
        


    # def get_one_model(self, bpm, offset_factor):
    #     """
    #     get one model using bpm and offset variables
    #     """
    #     samples_per_beat = int(1 / (bpm * 60) * self.rate)
    #     offset = -int(samples_per_beat * offset_factor)
    #     high_bound = int(max(self.peaks[:, 0]) + (2 * self.rate))

    #     # fraction of one beat
    #     # aka quarternote, sixteenthnote, eighthnote, sixteenthnote
    #     fractions = [1.0, 1/4, 1/2, 1/4]
    #     eighths = [int(i * self.rate / 4) for i in range(4)]

    #     model_beats = []
    #     for i in range(offset, high_bound, samples_per_beat):
    #         for j in range(4):
    #             # each model beats as [index, beat type]
    #             model_beats.append([i + eighths[j], fractions[j]])

    #     return np.asarray(model_beats)


    # def model_cost(self, model, weights):
    #     """
    #     cost function for optimizing model
    #     """
    #     # match peaks to model beats, starting with highest peaks
    #     peak_model_match = [] # [ [peak, model_match] ]
    #     for peak in self.peaks:
    #         diffs = (model[:,0] - peak[0])
    #         min_ind = diffs.argmin()
    #         model_match = model[min_ind]
    #         model = np.reshape(np.delete(model, min_ind, 0), (-1, 2))
    #         peak_model_match.append(np.asarray([peak, model_match]))

    #     costs = []
    #     for match in peak_model_match:
    #         # difference between sample indexes
    #         diff = np.abs(match[0][0] - match[0][1])
    #         # amplitude times difference between scaled peak and beat weight
    #         diff *= np.abs(match[0][1] - NpOps.sigmoid(weights[match[1][1]]))
    #         costs.append(diff)
    #     return tf.cast(np.mean(costs), tf.float32)






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