from autodrummer import *
import re
import time
import gc
import random as rd

class TestAutodrummer():
    """
    test autodrummer for a set with known bpms
    """

    def __init__(self, learning_rate=0.001, passes_per_model=10, num_models=5):
        print(learning_rate, passes_per_model, num_models)
        training_data = self.get_training_data()
        processed_data = []
        for d in training_data:
            filename = re.sub(r" ", "", d[0])
            rec = Recording(
                source='training_sources/' + filename + ".wav",
                name=filename
            )

            drummer = AutoDrummer(rec)
            drummer.configure(
                learning_rate=learning_rate,
                num_models=int(num_models),
                passes_per_model=int(passes_per_model)
            )
            model = drummer.run(True, False, False)

            processed_data.append({
                "name": filename, 
                "expected": float(d[1]), 
                "actual": model['bpm'],
                "cost": model['cost'],
                "pass": model['pass']
            })
        factors = []
        for d in processed_data:
            factor = d['actual']/d['expected']
            factors.append([d['name'], factor])
            print(d['name'], "factor:", factor)
            print("    cost", d['cost'], "pass", d['pass'])
            print("    expected: {0}, actual: {1:.4f}".format(
                d['expected'], d['actual']))
        close = []
        not_close = []
        for i in factors:
            if 0.4 <= i[1] <= 0.6:
                factr = round(i[1] * 4) / 4
            else:
                factr = round(i[1])
            error = abs(i[1] - factr)
            if (
                error < 0.05 * factr
            ):
                close.append(str(i[0]) + " " + str(i[1]))
            else:
                not_close.append(str(i[0]) + " " + str(i[1]))
        print("")
        print("Not Close:")
        for i in not_close: print("   ", i)
        print("Close:")
        for i in close: print("   ", i)
        print("{0}/{1}: {2:.4f}%".format(len(close), len(factors), (100 * len(close)/len(factors))))
        self.close = len(close)        


    def get_training_data(self):
        """
        read training_data.csv.
        returns [ [filename, bpm], ... ]
        """
        with open(dirname(__file__) + "training_data.csv", "r") as f:
            data = [
                [j.strip() for j in i.split(",")][:2] for i in f.readlines()
                if i != "\n"
            ]
        return np.asarray(data)


    def get_close(self):
        return self.close


class MetaTestAutodrummer():
    """
    test which number of passes and models is best
    """

    def __init__(self, outfile):
        self.outfile = outfile + '.tsv'
        passes = [5, 8, 10, 15, 20, 50, 75, 100, 200]
        # self.run(passes, 5)
        self.view()



    def get_training_data(self):
        """
        read training_data.csv.
        returns [ [filename, bpm], ... ]
        """
        with open(dirname(__file__) + "training_data.csv", "r") as f:
            data = [
                [j.strip() for j in i.split(",")][:2] for i in f.readlines()
                if i != "\n"
            ]
        return np.asarray(data)


    def run(self, passes, num_models):
        training_data = self.get_training_data()
        
        # f = open('outdata.tsv', 'w')
        # f.close()
        for i in range(10):
            song = training_data[rd.randint(0, len(training_data))]
            filename = re.sub(r" ", "", song[0])
            rec = Recording(
                source='training_sources/' + filename + ".wav",
                name=filename
            )

            drummer = AutoDrummer(rec)
            for p in passes:
                drummer.configure(
                    num_models=num_models,
                    passes_per_model=p
                )
                model = drummer.run(False, False, False)

                ratio = model['bpm'] / float(song[1])

                close = False
                if 0.4 <= ratio <= 0.6:
                    factr = round(ratio * 4) / 4
                else:
                    factr = round(ratio)
                weighted_error = abs(ratio - factr) / factr
                if weighted_error < 0.05:
                    close = True
                
                with open(self.outfile, 'a') as f:
                    f.write(
                        "\t".join([str(p), str(weighted_error)]) + "\n"
                    )
            del rec
            del drummer
            del model
            del f
            gc.collect()
     

    def view(self):
        with open('outdata.tsv', 'r') as f:
            data = f.readlines()
        with open(self.outfile, 'r') as f:
            data += f.readlines()
        data = [i.strip() for i in data if len(i.strip()) > 0]
        data = np.array([[int(i), float(j)] for i, j in [k.split("\t") for k in data]])
        passes_data = {}
        for line in data:
            try:
                passes_data[line[0]].append(line[1])
            except KeyError:
                passes_data[line[0]] = [line[1]]
        for k, v in passes_data.items():
            passes_data[k] = sum(v) / len(v)
        for k, v in passes_data.items():
            print("{0} passes, {1:.4f} avg error".format(k, v))
        plt.scatter(list(passes_data.keys()), (list(passes_data.values())))
        plt.show()
        



def test_autodrummer_main():
    if "meta" in sys.argv:
        MetaTestAutodrummer('out2')
    else:
        TestAutodrummer(*[float(i) for i in sys.argv[1:]])


if __name__ == "__main__":
    test_autodrummer_main()