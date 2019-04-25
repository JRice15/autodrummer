import pyaudio
import wave
import sys

CHUNK = 1024
print("record time (seconds): ", end="")
RECORD_SECONDS = int(input())
CHANNELS = 2
RATE = 44100
FORMAT = pyaudio.paInt16
print("output file name: ", end="")
WAVEFILE_OUTPUT = input() + ".wav"

print("begin recording? [y/n]: ", end="")
begin = input()
if begin == "y":

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    # open stream (2)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index=0,
                    input=True)

    print("* recording at {0} to file {1}".format(p.get_device_info_by_index(0)['name'], WAVEFILE_OUTPUT))

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()  
    p.terminate()

    wf = wave.open(WAVEFILE_OUTPUT, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()