import sys
import numpy as np
import tensorflow as tf
import os
import random as rd

from os.path import dirname

global relativism_dir
relativism_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.append(relativism_dir)


from src.analysis import *
from src.recording_obj import Recording
from src.sampler import *
from src.input_processing import *
from src.output_and_prompting import *



class AutoDrummer(Analysis):
    """
    init with rec, configure optimizer if nescessary, then run()
    """

    def __init__(self, rec):
        super().__init__(rec)

        self.set_frame_fractions(1/20, 1/100)

        # get peaks from waveform
        frames = self.get_frames_mono()
        # self.plot(frames, title="Frames")
        all_peaks = self.find_peaks(frames)
        # self.plot(all_peaks, plot_type="scatter", title="All Peaks")

        self.peaks = self.filter_peaks(all_peaks)
        # self.plot(self.peaks, plot_type="scatter", title="Filtered Peaks")

        # define optimizer behavior
        self.learning_rate = 0.001
        self.num_models = 20
        self.passes_per_model = 200
        self.base_peak_level = -0.0002 # higher negative value penalizes missed peaks more. careful changing much


    def configure(self, learning_rate=None, num_models=None, passes_per_model=None, 
            base_peak_level=None):
        """
        learning rate, num_models, passes_per_model, base_peak_level
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if num_models is not None:
            self.num_models = num_models
        if passes_per_model is not None:
            self.passes_per_model = passes_per_model
        if base_peak_level is not None:
            self.base_peak_level = base_peak_level


    def format_peaks(self, peaks):
        # scale to 0-1 range
        high = np.percentile(peaks[:,1], 93)
        peaks[:,1] = peaks[:,1] / high
        peaks[:,1][peaks[:,1] > 1] = 1

        high = NpOps.column_max(peaks, 0)
        low = NpOps.column_min(peaks, 0)
        base_vals = np.empty((int((high - low)/self.frame_step)))
        base_vals.fill(self.base_peak_level)
        base_inds = np.arange(low, high, self.frame_step)
        base_peaks = NpOps.join_channels(base_inds, base_vals)
        peaks = NpOps.set_indexes(base_peaks, peaks)

        # self.plot(peaks, plot_type="scatter", title="Formatted Peaks")

        # make tensor
        formatted_peaks = tf.constant(
            peaks,
            dtype=tf.float32,
            name="tensor_peaks"
        )

        # peaks as (sample_index, scaled slope), sorted by slope
        return formatted_peaks


    def run(self, show_passes=True, plot_final=False, save_run=True):
        """
        run tensorflow model
        """

        # tf constants
        self.tf_peaks = self.format_peaks(self.peaks)
        self.tf_pi = tf.constant(np.pi, dtype=tf.float32, name="pi")
        beat_factors = tf.constant( 
            # relative lengths of beats [beat, hb, qb]
            [1.0, 1/2, 1/4],
            dtype=tf.float32,
            name="beat_factors"
        )

        # tf variables
        bpm_factor = self.tf_sigmoid(
            tf.Variable(tf.random.uniform([1], 0, 1), dtype=tf.float32,  name='bpm')
        )
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
        bpm = 120 * bpm_factor + 60
        samples_per_beat = tf.multiply(1 / bpm * 60, self.rate, name="samples_per_beat")
        beat_lengths = tf.multiply(beat_factors, samples_per_beat, name="beat_lengths")
        offset_length = tf.multiply(samples_per_beat, offset_factor, name="offset_length")

        # calculating cost
        cost = self.model_cost(beat_lengths, offset_length, beat_wave_weights)

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate,
            name="optimizer"
        )
        optimization = optimizer.minimize(cost)

        # running
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            if save_run:
                file_writer = tf.summary.FileWriter(
                    dirname(os.path.abspath(__file__)) + "/logs", 
                    sess.graph
                )
        
            best_models = []
            
            for m in range(self.num_models):
                # reset variables
                sess.run(init)
                best_run = None

                # initial variables
                run_cost = sess.run(cost)
                run_bpm = sess.run(bpm)[0]
                run_offset = sess.run(offset_factor)[0]
                run_weights = sess.run(beat_wave_weights)
                print("model {0}".format(m))
                initials = (-1, run_cost, run_bpm, run_offset, run_weights)

                # run passes
                for p in range(self.passes_per_model):
                    sess.run(optimization)
                    run_cost = sess.run(cost)
                    run_bpm = sess.run(bpm)[0]
                    run_offset = sess.run(offset_factor)[0]
                    run_weights = sess.run(beat_wave_weights)
                    if show_passes:
                        self.display_pass(p, run_cost, run_bpm, run_offset, run_weights)
                    if best_run is None or run_cost < best_run['cost']:
                        best_run = {
                            "cost": run_cost,
                            "bpm": run_bpm,
                            "offset": run_offset,
                            "weights": run_weights,
                            "pass": p
                        }
                best_models.append(best_run)
                if show_passes:
                    self.display_pass(*initials)

        very_best_model = {'cost': 10000}
        for i in best_models:
            if i['cost'] < very_best_model['cost']:
                very_best_model = i
            if show_passes:
                self.display_model(i)

        print("")
        self.display_model(very_best_model)
        if plot_final:
            self.plot_model(very_best_model)
        return very_best_model


    def tf_sigmoid(self, arr):
        return 1 / (1 + 7.39 * tf.exp(-4 * arr))


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
        return tf.math.divide(
            tf.reduce_sum(
                tf.abs(
                    (self.tf_peaks[:,1] - full_cost_wave) * self.tf_peaks[:,1],
                    name="abs_point_costs"
                ),
                name="total_cost"
            ),
            self.sec_len,
            name="cost_per_second"
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
                    (indexes - peak_index) * self.tf_pi / wavelength
                ),
                16
            ),
            name=name
        )
        
        
    # representing
    def plot_model(self, model):
        indexes = np.arange(self.start, self.end, self.frame_step)
        beat_factors = np.array([1.0, 0.5, 0.25])

        samples_per_beat = 1 / model['bpm'] * 60 * self.rate
        beat_lengths = beat_factors * samples_per_beat
        offset_length = samples_per_beat * model['offset']

        wave_beat = self.wave_distribution_model(
            indexes=indexes,
            wavelength=beat_lengths[0],
            peak_index=offset_length,
            wave_weight=model['weights'][0],
            name="whole_beat_wave"
        )
        wave_halfbeat = self.wave_distribution_model(
            indexes=indexes,
            wavelength=beat_lengths[1],
            peak_index=offset_length,
            wave_weight=model['weights'][1],
            name="half_beat_wave"
        )
        wave_quarterbeat = self.wave_distribution_model(
            indexes=indexes,
            wavelength=beat_lengths[2],
            peak_index=offset_length,
            wave_weight=model['weights'][2],
            name="qrtr_beat_wave"
        )
        full_cost_wave = wave_beat + wave_halfbeat + wave_quarterbeat
        with tf.Session() as sess:
            wave = sess.run(full_cost_wave)
        plt.plot(indexes, wave)
        plt.scatter(self.peaks[:,0], self.peaks[:,1])
        plt.show()

    def display_pass(self, p, cost, bpm, offset, weights):
        """
        representation for one pass of the optimizer
        """
        print("  pass {0:04d}, cost {1:.5f}, bpm {2:.5f}, offset_factor {3:.5f}, weights: {4:.5f} {5:.5f} {6:.5f}".format(
            p, cost, bpm, offset, weights[0][0], weights[1][0], weights[2][0]))

    def display_model(self, model):
        """
        representation for one model created by optimizer
        """
        print("cost", model['cost'], "pass", model['pass'])
        print("    bpm", model['bpm'])
        print("    offset", model['offset'])
        print("    weights", model['weights'][0][0], model['weights'][1][0], model['weights'][2][0])



    def play_bpm(self, bpm):
        samples_per_beat = 60 / bpm * self.rate
        arr = np.zeros((int(samples_per_beat * 4), 2))
        for i in range(4):
            arr[int(i * samples_per_beat)] = 1
        Recording(arr, name="bpmtest").playback(0)










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
    a = Recording(source='training_sources/bpm_83.wav', name='test')
    drummer = AutoDrummer(a)
    model = drummer.run(True, True)
    # a.playback()
    # drummer.play_bpm(model['bpm'])









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