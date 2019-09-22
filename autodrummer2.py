import pyaudio
import wave
import sys
import sounddevice as sd
import numpy
import time
import soundfile as sf
import re
import random as rd
import math
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

import tensorflow as tf



class AutoDrummer(Analysis):

    def __init__(self, rec):
        super().__init__(rec)

        self.set_frame_fractions(1/20, 1/100)

        self.frames = self.get_frames_mono()
        peaks = self.find_peaks(self.frames)

        # tensor constants
        self.peaks = tf.convert_to_tensor(
            self.filter_peaks(peaks),
            dtype=tf.float32,
            name="peaks"
        )
        self.rate = tf.Constant(self.rate, dtype=tf.float32, name="sample rate")

        # run modelling
        self.model()
    

    def read_model(self):
        pass


    def model(self):
        """
        run tf model
        """
        self.gradient_step = 0.01
        self.num_models = 10
        self.passes_per_model = 1000


        model = []
        while True:
            model.append()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
        
            best_model = []
            for _ in range(self.num_models):
                result = sess.run()


    def get_one_model(self):
        """
        get one model using bpm and offset variables
        """
        bpm = tf.Variable(tf.random.uniform(60, 120), dtype=tf.float32,  name='bpm')
        offset_factor = tf.Variable(tf.random.uniform(0, 1), dtype=tf.float32, name="offset factor")
        offset = offset_factor * self.rate
        samples_per_beat = 1 / (bpm * 60) * self.rate

        for i in range(-offset, max(self.peaks[:, 0]))


    def model_cost(self, model)







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