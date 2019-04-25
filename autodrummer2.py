import pyaudio
import wave
import sys
import sounddevice as sd
import numpy
import time
import soundfile as sf
import re
import math


RATE = 44100


def main():
    """
    real thalamus hours
    """

    global RATE
    FRACTION = 1/20  # 0.05

    mode, record_time, filename, playback_yn, device_ind = command_line_init()

    # also covers playback
    recording = init_and_process(mode, record_time, filename, playback_yn, device_ind)

    peaks = find_peaks(recording, RATE, FRACTION, record_time)



def command_line_init():
    """
    analyze command line for initialization flags
    """
    mode = None
    record_time = None
    filename = None
    playback_yn = None
    device_ind = None

    for i in sys.argv[1:]:
        if re.fullmatch(r"^mode=.+", i) or re.fullmatch(r"^m=.+", i):
            mode = re.sub(r".+=", "", i)
        elif re.fullmatch(r"^duration=.+", i) or re.fullmatch(r"^d=.+", i):
            record_time = int(re.sub(r".+=", "", i))
        elif re.fullmatch(r"^filename=.+", i) or re.fullmatch(r"^f=.+", i):
            filename = re.sub(r".+=", "", i)
        elif re.fullmatch(r"^playback=.+", i) or re.fullmatch(r"^p=.+", i):
            playback_yn = re.sub(r".+=", "", i)
        elif re.fullmatch(r"^device=.+", i) or re.fullmatch(r"^d=.+", i):
            device_ind = int(re.sub(r".+=", "", i))
        else:
            print("Unrecognized command line flag: '" + i +"'. Ignoring...")

    return mode, record_time, filename, playback_yn, device_ind


def init_and_process(mode, record_time, filename, playback_yn, device_ind):
    """
    fill in initialization via input
    get recording by mode
    call get_help/playback if asked
    """

    valid_modes = ("live record (r)", "read .wav file (f)", "help (h)")
    if mode == None:
        print("\nAvailable modes:")
        for i in valid_modes: print("* {0}".format(i))
        print("Select mode: ", end="")
        mode = input()

    # Record Mode
    if mode.lower() in ("r", "rec", "record", "live", "live record", "(r)"):
        print("\n* Record mode\n")
        if device_ind == None:
            print("{0} devices found".format(len(sd.query_devices())))
            print("Devices by index:")
            print(sd.query_devices())
            print("\nEnter desired recording device index: ", end="")
            device_ind = int(input())
        if record_time == None:
            print("\nSelect recording duration (seconds): ", end="")
            record_time = int(input())
        device_name = sd.query_devices()[device_ind]['name']
        recording = record_live(RATE, device_ind, device_name, record_time)

    # File Mode
    elif mode.lower() in ("f", "file", "read", ".wav file", "wav", "wav file",\
            "read .wav file", "(f)"):
        print("\n* File mode\n")
        recording = None
        while type(recording) != numpy.ndarray:
            if filename == None:
                print("Enter file full name: ", end="")
                filename = input()
            recording = read_file(filename)
            if type(recording) != numpy.ndarray:
                print("\nFile '{0}' unable to be read".format(filename))
                filename = None
                print("Please select a '.wav' file")
        record_time = int(len(recording) / RATE)

    # Help
    elif mode.lower() in ("h", "help", "(h)"):
        get_help()

    # Unknown mode
    else:
        print("\nInvalid mode '{0}'. Valid modes are:".format(mode))
        for i in valid_modes: print("* {0}".format(i))
        print("\nExiting...\n")
        sys.exit()

    return recording


def record_live(RATE, device_ind, device_name, record_time):
    """
    record -- but get this -- Live!
    """

    print("\nRecord? [y/n]: ", end="")
    cont = input()
    if cont not in ("Y", "y", "yes", "Yes"):
        print("\nExiting...\n")
        sys.exit()
    time.sleep(0.4)
    print("* Recording at input {0} ({1}) for {2} seconds".format(device_ind, \
        device_name, record_time))
    recording = sd.rec(int(record_time * RATE), RATE, device=device_name)
    sd.wait()
    print("Finished recording")
    return recording


def read_file(filename):
    global RATE
    try:
        recording, RATE = sf.read(filename)
        print("Sound file '{0}' read successfully".format(filename))
        return recording
    except RuntimeError:
        return None


def playback(recording, playback_yn):
    if playback_yn == None:
        print("\nPlayback before analyzing? [y/n]: ", end="")
        playback_yn = input()
    if playback_yn in ("Y", "y", "yes", "Yes"):
        time.sleep(0.5)
        print("\n* Playback")
        sd.play(recording, RATE)
        sd.wait()
        print("Finished playback")


def find_peaks(recording, RATE, FRACTION, record_time):
    """
    * Calculate peaks in waveform
    * Returns _____
    """
    frames_frac = int(RATE * FRACTION) # 20th of second fractions
    iter_frames = int(frames_frac / 5) # 441. Iterated at 100ths of seconds


    """ take and average data from recording into snippets """
    snippet_averages = [] # start frame, avg amplitude
    for i in range(0, len(recording) - frames_frac, iter_frames): 
        snippet = recording[i:i + frames_frac]
        snippet = numpy.take(snippet, range(len(snippet)))
        snippet = [abs(j) for j in snippet]
        snippet_averages.append([i, sum(snippet) / len(snippet)])


    """ get highest non-overlapping peaks within a given 20th of a second"""
    snippet_averages = sort(snippet_averages, 1, None, True)
    
    ref_ind = 0
    while ref_ind < len(snippet_averages):
        del_ind = ref_ind + 1
        ref_ind_frame = snippet_averages[ref_ind][0]
        while del_ind < len(snippet_averages):
            if snippet_averages[del_ind][0] - (5 * iter_frames) < ref_ind_frame < snippet_averages[del_ind][0] + (5 * iter_frames):
                del snippet_averages[del_ind]
            else:
                del_ind += 1
        ref_ind += 1

    print(snippet_averages[:30])













def sort(the_list, ind, ind2, rev):
    if ind != None:
        if ind2 != None:
            return sorted(the_list, key=lambda i: i[ind][ind2], reverse=rev)
        return sorted(the_list, key=lambda i: i[ind], reverse=rev)
    else:
        return sorted(the_list, reverse=rev)



def get_help():
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
        get_help()
    else:
        main()