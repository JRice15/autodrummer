import pyaudio
import wave
import sys
import sounddevice as sd
import numpy
import time
import soundfile as sf
import re


RATE = 44100


def main():

    #try:
        
        # Initialization
        sd.default.channels = 2

        print("\n*** Welcome to AutoDrummer ***\n")
        
        mode, record_time, filename, playback_yn, device_ind = initialize()

        # Modes
        recording, mode, record_time, filename, playback_yn, device_ind = \
            process_mode(mode, record_time, filename, playback_yn, device_ind)

        #Playback
        playback(recording, playback_yn)

        start_time = time.time()

        # Average Fractions
        pos_diffs, total_avg = averages(recording, RATE, record_time)
        
        # Find Possible Beats
        poss_beats = possible_beats(pos_diffs, record_time, total_avg)

        # Find Lengths of Beats
        beat_lens = beat_lengths(poss_beats)

        # Find lengths of most likely beats
        tempos = analyze_tempo(beat_lens)

        # Find Pattern of Beat Lens
        pattern = find_pattern(beat_lens, tempos)
        pattern = pattern[4:]

        # find measure length from the pattern
        best_measure_model = group_rhythms(pattern, tempos)
        measure_len = best_measure_model[0]

        # get type of beats from measure len
        beat_approxs = type_beats(tempos, measure_len)

        # get final pattern model
        model, rhythm_fracs = create_pattern(best_measure_model, beat_approxs, pattern)

        # get end pattern
        end_model = end_pattern(best_measure_model, beat_lens)

        # Playback tempo
        play_tempo_basic(model, measure_len, start_time)

        end_time = time.time()
        total_time = end_time - start_time
        print("\nRuntime: {0} seconds".format(total_time))
        print("*** Complete ***\n")
    
    #except:
        #print("error")


def initialize():
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


def process_mode(mode, record_time, filename, playback_yn, device_ind):
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
        recording = record(RATE, device_ind, device_name, record_time)

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
        helper()

    # Unknown mode
    else:
        print("\nInvalid mode '{0}'. Valid modes are:".format(mode))
        for i in valid_modes: print("* {0}".format(i))
        print("\nExiting...\n")
        sys.exit()

    return recording, mode, record_time, filename, playback_yn, device_ind


def record(RATE, device_ind, device_name, record_time):
    print("\nRecord? [y/n]: ", end="")
    cont = input()
    if cont not in ("Y", "y", "yes", "Yes"):
        print("\nExiting...\n")
        sys.exit()
    time.sleep(1)
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


def averages(recording, RATE, record_time):
    print("\n* Analyzing Data\n")
    ### Finding averages of fractions
    fraction = 100
    interval = int(RATE / fraction) # cuts each second into fraction parts
    count = 0
    list_avgs = []
    for low in range(0, RATE * record_time, interval):
        high = low + interval
        snippet = recording[low:high]
        values = numpy.take(snippet, range(len(snippet)))
        abs_vals = [abs(i) for i in values]
        avg = sum(abs_vals)/len(abs_vals)
        list_avgs.append(avg)
        # print("\nplaying snippet", count)
        # sd.play(recording[low:high])
        # sd.wait()
        # print("avg from frame", low,  ":", avg)
        count += 1
    total_avg = sum(list_avgs) / len(list_avgs)
    print("Average amplitude: ", total_avg)

    ### Finding rates of change between fractions
    diff = [list_avgs[i+1] - list_avgs[i] for i in range(len(list_avgs) - 1)]
    enum_diff = list(enumerate(diff, 0))
    pos_diffs = [i for i in enum_diff if i[1] > 0]

    return pos_diffs, total_avg


def possible_beats(pos_diffs, record_time, total_avg):
    ### Identifying possible beats
    top_n = 3 * record_time

    sorted_diff = selection_sort(pos_diffs, top_n, 1) # sort by values
    poss_beats = [i for i in sorted_diff if i[1] > total_avg]
    poss_beats = [i for i in poss_beats if i[1] > (1/50) * poss_beats[0][1]]
    # for i in poss_beats:
    #     for j in poss_beats:
    #         if j != i:
    #             if i[0] - 5 <= j[0] <= i[0] + 5:
    #                 poss_beats.remove(j)
    # print("\npossible beats: ", poss_beats)

    # for i in poss_beats: #playing possible beats
    #     time.sleep(0.5)
    #     print("\nplaying beat", i)
    #     temp_low = (i[0]) * interval
    #     temp_high = (i[0] + 10) * interval
    #     sd.play(recording[temp_low:temp_high])
    #     sd.wait()

    return poss_beats


def beat_lengths(poss_beats):
    beat_sort = selection_sort(poss_beats, len(poss_beats), 0) # sort by indices
    beat_lens = []
    for i in range(len(beat_sort) - 1, 0, -1): # calculate time diffs
        temp_diff = beat_sort[i-1][0] - beat_sort[i][0]
        beat_lens.append(temp_diff)
    for i in range(len(beat_lens) - 1): # consolidate split beats
        if beat_lens[i] <= 5:
            beat_lens[i+1] += beat_lens[i]
    beat_lens = [i for i in beat_lens if i > 5]
    print("\nBeat lens:", beat_lens)
    return beat_lens


def analyze_tempo(beat_lens):
    tempo_matches = []
    for i in range(len(beat_lens)): # find which lengths have most similar beats
        tmatches_i = 0
        running_tot = 0
        intvl = [(beat_lens[i]) - 5, (beat_lens[i]) + 5]
        for j in range(len(beat_lens)):
            running_tot += beat_lens[j]
            if intvl[0] <= running_tot <= intvl[1]:
                tmatches_i += 1
                running_tot = 0
            elif running_tot > intvl[1]:
                running_tot = 0
        tempo_matches.append(tmatches_i)
    beats_tempos = [(beat_lens[i], tempo_matches[i]) for i in \
            range(len(beat_lens))]
    likely_tempo = selection_sort(beats_tempos, len(beats_tempos), 1) # (tempo, count/weight/likeliness)
    print("Likely tempos:", likely_tempo)
    tempo_groups = [likely_tempo[0]]
    for i in range(1, len(likely_tempo)): # find representative beat lengths
        intvl = [likely_tempo[i][0] - 5, likely_tempo[i][0] + 5]
        count = 0
        for j in tempo_groups:
            if not (intvl[0] <= j[0] <= intvl[1]):
                count += 1
        if count == len(tempo_groups):
            tempo_groups.append(likely_tempo[i])
    print("Tempo groups:", tempo_groups)
    weighted_tempos = []
    for i in tempo_groups: # for groups in interval of most common, find average
        intvl = [i[0] - 5, i[0] + 5]
        vals_in_intvl = [j[0] for j in likely_tempo if (intvl[0] <= j[0]<= \
                intvl[1])]
        weighted = sum(vals_in_intvl) / len(vals_in_intvl)
        weighted_tempos.append(weighted)
    indices_to_rm = []
    for i in range(len(weighted_tempos)): # remove overlapping groups after weighting
        for j in weighted_tempos[:i]:
            if j - 5 <= weighted_tempos[i] <= j + 5:
                indices_to_rm.append(i)
    for i in indices_to_rm:
        weighted_tempos[i] = "rmv"
    while "rmv" in weighted_tempos:
        weighted_tempos.remove("rmv")
    print("Weighted:", weighted_tempos)
    return weighted_tempos


def find_pattern(beat_lens, tempos):
    rhythms = []
    for i in beat_lens: # match beat lens to their tempo group
        dist = None
        for j in range(len(tempos)):
            intvl = [tempos[j] - 5, tempos[j] + 5]
            if intvl[0] <= i <= intvl[1]:
                if (dist == None) or (dist[1] > abs(tempos[j] - i)):
                    dist = [j, abs(tempos[j] - i)]
        if dist == None:
            rhythms.append(-1)
        else:
            rhythms.append(dist[0])
    while -1 in rhythms[0:3]:
        rhythms.remove(-1)
    print("Beat lens:", beat_lens)
    print("Rhythms:", rhythms)
    return rhythms


def group_rhythms(rhythms, weighted):
    poss_measure_lens = []
    weightedt = [i for i in weighted]
    weightedt.insert(0, 0) # find all combos of adding up-to 5 tempos together
    for b in [1, 2, 3, 4]:
        for i in weightedt:
            for j in weightedt:
                for k in weightedt:
                    for l in weighted:
                        for m in weighted:
                            num = int(b * (i + j + k + l + m))
                            if num <= 410: # more than 60 bpm
                                poss_measure_lens.append(num)
    poss_measure_lens = list(set(poss_measure_lens))
    print("Possible measure lengths ({0}):".format(len(poss_measure_lens)), \
            poss_measure_lens[0:10], "...", poss_measure_lens[-10:])

    longest_model = []
    for start in range(min(len(rhythms), 8)):
        for i in poss_measure_lens: # find which tempo best models measures of the music
            temp_longest = [0, 0] # len, ind
            temp_len_measures = [0, start] # len, ind
            temp_len_fracs = 0
            for j in range(start, len(rhythms)):
                if rhythms[j] == -1:
                    if temp_longest[0] < temp_len_measures[0]:
                        temp_longest = [k for k in temp_len_measures]
                    temp_len_measures[0] = 0
                    temp_len_measures[1] = j+1
                    temp_len_fracs = 0
                else:
                    temp_len_fracs += weighted[rhythms[j]]
                if (i - 5 <= temp_len_fracs <= i + 5):
                    temp_len_measures[0] += 1
                    temp_len_measures.append(abs(i - temp_len_fracs))
                    temp_len_fracs = 0
                elif (temp_len_fracs > i + 5):
                    if temp_longest[0] < temp_len_measures[0]:
                        temp_longest = temp_len_measures
                    temp_len_measures[0] = 0
                    temp_len_measures[1] = j+1
                    temp_len_fracs = 0
            if temp_longest[0] < temp_len_measures[0]:
                temp_longest = temp_len_measures
            temp_longest.insert(0, i)
            longest_model.append(temp_longest)
    for i in longest_model:
        i[1] = i[0] * i[1]
    longest_model = selection_sort(longest_model, len(longest_model), 1) # sort by length of match
    longest_model = [i for i in longest_model if i[1] > 0.75 * \
            longest_model[0][1]]
    weighted_model = [] # find which model is most accurate
    for i in longest_model:
        sub_list = i[0:3]
        sub_list.append(sum(i[3:]) / len(i[3:]))
        weighted_model.append(sub_list)
    print("weighted model:", weighted_model[:10], "...", weighted_model[-10:]) # tempo, len, ind, precision
    sort_models = selection_sort(weighted_model, len(weighted_model), 3)
    print(sort_models)
    best_model = sort_models[-1]
    print("best model:", best_model)
    return best_model


def type_beats(tempos, measure_len): # beats = weighted tempos
    measure_fracs = [ i/measure_len for i in tempos]
    print("tempos, measure len", tempos, measure_len)
    print("measure fracs", measure_fracs)
    real_beats =    ((1, "w"), # whole
                    (1/2, "h"), # half
                    (1/3, "t"), # third
                    (1/4, "q"), # quarter
                    (1/8, "e"), # eighth
                    (1/12, "tw"), # twelvth
                    (1/16, "s"), # sixteenth
                    (1.5, "dw"), # dotted whole
                    (1.5/2, "dh"), # dotted half (three quarters)
                    (1.5/4, "dq"), # dotted quarter (three eighths)
                    (1.5/8, "de")) # dotted eighth (three sixteenths)
    print("real beats", real_beats)
    beat_approxs = []
    for m in range(len(measure_fracs)):
        closeness = []
        for r in real_beats:
            closeness.append(abs((measure_fracs[m] / r[0]) - 1))
        beat = closeness.index(min(closeness))
        beat_approxs.append(real_beats[beat]) # actual len, type, fraction of measure
    print("tempos", tempos)
    print("measure fracs", measure_fracs)
    print("beat approxs", beat_approxs)
    return beat_approxs


def create_pattern(best_measure_model, beat_approxs, pattern):
    rhythm_fracs = []
    for i in pattern:
        if i == -1:
            rhythm_fracs.append(-1)
        else:
            rhythm_fracs.append(beat_approxs[i][0])
    print("rhythm fracs", rhythm_fracs)

    start = best_measure_model[2] # get highest power of two repetitions it models
    for i in [64, 32, 16, 8, 4, 2, 1]:
        if i <= best_measure_model[1] / best_measure_model[0]:
            end_val = best_measure_model[1] / best_measure_model[0]
            break

    model = [0] # len in measures, values
    temp_meas = []
    for i in range(start, len(rhythm_fracs)): # get model to play, in hundreths of seconds
        if rhythm_fracs[i] == -1:
            break
        else:
            temp_meas.append(rhythm_fracs[i])
            if (0.95 <= sum(temp_meas) <= 1.05):
                model[0] += 1
                model = model + temp_meas
                temp_meas = []
            elif (sum(temp_meas) > 1.05):
                break
        if model[0] == end_val:
            break
    model = [i for i in model[1:]]
    print("model", model)
    return model, rhythm_fracs


def end_pattern(best_measure_model, beat_lens):
    print("\n\n")
    print(best_measure_model, beat_lens)

    start = best_measure_model[2]
    length = best_measure_model[1]
    measures = length / best_measure_model[0]
    # get what index is right after the x number of measures and find out
    # length of rest of selection


def play_tempo_basic(model, measure_len, start_time):
    print("a")
    global RATE
    snap, RATE = sf.read("snap.wav")

    # model_sum = 0
    # adding = True
    # i = 0
    # elapsed = 100 * (time.time() - start_time)
    # while adding:
    #     model_sum += model[i]
    #     i += 1
    #     if i >= len(model):
    #         i = 0
    #     if model_sum > elapsed:
    #         adding = False
    #         time.sleep(max(0.01 * (model_sum - elapsed - 3), 0))
    #         for i in model[i:]:
    #             print("." * int(i/2))
    #             snap_temp = snap[:int((RATE * 0.01 * i) - (RATE * 0.13346))]
    #             sd.play(snap_temp)
    #             sd.wait()
    for _ in range(16):
        print("")
        for i in model:
            print("." * int(i * measure_len / 2))
            snap_temp = snap[:int((RATE * 0.01 * i * measure_len) - (RATE * 0.13346))]
            sd.play(snap_temp)
            sd.wait()




def selection_sort(unsorted, top_n, ind):
    sortedd = []
    unsorted2 = [i[ind] for i in unsorted]
    for _ in range(top_n):
        highest_ind = unsorted2.index(max(unsorted2))
        sortedd.append(unsorted[highest_ind])
        unsorted2[highest_ind] = 0
    return sortedd


def helper():
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
        helper()
    else:
        main()