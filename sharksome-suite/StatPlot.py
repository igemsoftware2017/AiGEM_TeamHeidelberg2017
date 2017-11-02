#Little tool to visualize intermediate log file data of MAWS runs with DNA.
#Requires an active matplotlib installation
import matplotlib.pyplot as plt
import argparse
#Enter filename here
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Path of the JOB_NAME_entropy.log file")
args = parser.parse_args()
if args.path:
    log = open(args.path)
    legendStrings = ["G3","G5","A3","A5","T3","T5","C3","C5","optimum"]
    entropies = [[],[],[],[],[],[],[],[],[]]
    energies = [[],[],[],[],[],[],[],[],[]]
    #Bad for performance or stability... But works
    for lcount, line in enumerate(log):
        if lcount < 4:
            index = lcount % 4
            entropy, energy = line.split(": ")[2:]
            entropy = entropy.split(" ")[0]
            entropies[index*2].append(float(entropy))
            entropies[index*2+1].append(float(entropy))
            energies[index*2].append(float(energy))
            energies[index*2+1].append(float(energy))
        else:
            index = lcount % 8
            entropy, energy = line.split(": ")[2:]
            entropy = entropy.split(" ")[0]
            entropies[index].append(float(entropy))
            energies[index].append(float(energy))
    for i in range(len(entropies[3])):
        pack = [(entropies[0])[i],(entropies[1])[i],(entropies[2])[i],(entropies[3])[i],(entropies[4])[i],(entropies[5])[i],(entropies[6])[i],(entropies[7])[i]]
        minIndex = min(enumerate(pack),key=lambda p: p[1])[0]
        entropies[8].append(entropies[minIndex][i])
        energies[8].append(energies[minIndex][i])
    plt.figure(1)
    plt.subplot(211)
    for i in range(9):
        plt.plot(entropies[i],label=legendStrings[i])
    plt.xlabel("# of steps")
    plt.ylabel("calculated minimal (entropy) score")
    plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.title("Score/Entropic trend")

    plt.subplot(212)
    for i in range(9):
        plt.plot(energies[i],label=legendStrings[i])
    plt.xlabel("# of steps")
    plt.ylabel("minimal energy of optimum")
    plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.title("Energetic trend")
    plt.show()

else:
    print('No path of entropy logfile provided! Add -p argument!')
