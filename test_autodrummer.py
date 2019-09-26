from autodrummer import *



class TestAutodrummer():


    def __init__(self):
        training_data = self.get_training_data()
        processed_data = []
        for d in training_data:
            filename = re.sub(r" ", "\ ", d[0])
            rec = Recording(
                source='training_sources/' + filename + ".wav",
                name=filename
            )

            drummer = AutoDrummer(rec)
            drummer.configure(0.01, 10, 1000)
            model = drummer.run(True, False, False)

            processed_data.append(
                [filename, float(d[1]), model['bpm'], model['cost']]
            )
        factors = []
        for d in processed_data:
            factors.append(d[1]/d[2])
            print(d[0], "factor:", d[1]/d[2])
            print("cost", d[3])
            print("    expected: {0}, actual: {1:.4f}".format(
                d[1], d[2]))
        close = []
        for i in factors:
            if abs(i % 2) < (0.1 * i):
                close.append(1)
            else:
                close.append(0)
        print("")
        print("close: {0}/{1}: {2:.4f}%".format(sum(close), len(close), (sum(close)/len(close))))
        


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




if __name__ == "__main__":
    TestAutodrummer()