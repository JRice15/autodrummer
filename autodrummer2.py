import pyaudio
import wave
import sys
import sounddevice as sd
import numpy
import time
import soundfile as sf
import re
import math

from os.path import dirname
musicdir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(musicdir + '/relativism')

from recording_obj import Recording
from sampler import *
from input_processing import *
from output_and_prompting import *



class Analysis():

    def __init__(self, rec=None):
        if rec is None:
            rec = Recording(parent=self)
        self.rec = rec
        self.arr = rec.arr
        self.mono_arr = [(i + j) / 2 for i, j in rec.arr]
        self.rate = rec.rate
        self.samp_len = rec.size_samps()
        self.average_amplitude = average(self.mono_arr)

        # frame variables
        self.frame_length = int(self.rate * 1/20) # 20th of second fractions
        self.frame_step = int(self.rate * 1/100) # 441. Iterated at 100ths of seconds


    def maybe_playback(self):
        p("Playback before analyzing? [y/n]")
        playback_yn = inpt()
        if playback_yn:
            self.rec.playback()


    def get_frames(self):
        """
        take and average data from recording into frames
        """
        info_block("Calculating frames")

        frames = [] # start index, avg amplitude
        for i in range(0, self.samp_len - self.frame_length, self.frame_step): 
            frame = self.mono_arr[ i : i + self.frame_length]
            frame = numpy.take(frame, range(len(frame)))
            frame_avg = average([abs(j) for j in frame])
            if frame_avg > self.average_amplitude:
                frames.append( (i, frame_avg) )

        return frames # (start index, avg amplitude)


    def find_peaks(self, frames):
        """
        get highest non-overlapping peaks within a given 20th of a second
        """
        info_block("Finding peaks")

        frames = [i for i in frames if i[1] > self.average_amplitude]
        sorted_frames = super_sort(frames, ind=1, high_to_low=True)
        
        print(sorted_frames[:20])

        i = 0
        while i < len(sorted_frames):
            del_ind = i + 1
            frame_ind = sorted_frames[i][0]
            while del_ind < len(sorted_frames):
                # if a smaller frame overlaps, delete it
                if (
                    sorted_frames[del_ind][0] - self.frame_length 
                    < frame_ind
                    < sorted_frames[del_ind][0] + self.frame_length
                ):
                    del sorted_frames[del_ind]
                else:
                    del_ind += 1
            i += 1

        in_order = super_sort(frames, ind=0)   
        for i in range(len(in_order) - 1):
            print(in_order[i + 1][0] - in_order[i][0])



class AutoDrummer(Analysis):

    def __init__(self, rec):
        super().__init__(rec)
        self.frames = self.get_frames()
        self.find_peaks(self.frames)







def average(the_list):
    """
    average items of a list
    """
    return sum(the_list) / len(the_list)


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
    a = Recording(source='/Users/user1/Desktop/CS/music/relativism/t.wav', name='test')
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