import sys, os

import numpy as np
import math
from scipy import interpolate
from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib as mpl

mpl.style.use('classic')

### setting parameter string GQME ###

# general constants
TIME_STEPS = 2000 # number of time steps
tau = 5              # propagation time step
au2ps = 0.00002418884254 # Conversion of attoseconds to atomic units
timeau = 12.409275
DT = tau * au2ps * timeau # time step in au
FINAL_TIME = TIME_STEPS * DT # final time
DOF_E = 2 # number of electronic states
DOF_E_SQ = DOF_E * DOF_E
DOF_N = 60 # number of nuclear DOF
HBAR = 1

### constants related to the model number ###
MODEL_NUM = 2 # model number

# setting parameters that based on the model number
BETA = 5 # inverse finite temperature beta = 1 / (k_B * T)
GAMMA_DA = 1 # diabatic coupling
if MODEL_NUM == 1:
    EPSILON = 1    # half of the energy gap between the donor and acceptor
    XI = 0.1       # friction coefficient, determines strength of e-n coupling
    OMEGA_C = 1    # cutoff frequency of the nuclear DOF
    OMEGA_MAX = 5  # maximum frequency of the nuclear DOF
elif MODEL_NUM == 2:
    EPSILON = 1
    XI = 0.1
    OMEGA_C = 2
    OMEGA_MAX = 10
elif MODEL_NUM == 3:
    EPSILON = 1
    XI = 0.1
    OMEGA_C = 7.5
    OMEGA_MAX = 36
elif MODEL_NUM == 4:
    EPSILON = 1
    XI = 0.4
    OMEGA_C = 2
    OMEGA_MAX = 10
elif MODEL_NUM == 6:
    EPSILON = 0
    XI = 0.2
    OMEGA_C = 2.5
    OMEGA_MAX = 12

### setting parameter string ###
PARAM_STR = "_Spin-Boson_Ohmic_TT-TFD_b%sG%s_e%s_t%.8f_"%(BETA, GAMMA_DA, EPSILON, DT)
PARAM_STR += "xi%swc%s_wmax%s_dofn%s_tf%.4f"%(XI, OMEGA_C, OMEGA_MAX, DOF_N, FINAL_TIME)

### specific constants to the memory kernel code ###
MEM_TIME = DT * TIME_STEPS # Either the memory time for straight calculation or
# for convergence
MEM_TIMESTEPS = int(MEM_TIME/DT)
FINAL_TIME_GQME = MEM_TIME + DT
FINAL_TIMESTEPS = int(FINAL_TIME/DT)

### setting parameter string GQME ###
PARAM_STR_GQME = PARAM_STR + "_mt%.4f_finalt%.4f"%(MEM_TIME, FINAL_TIME_GQME)

### variables ###
timeVec = np.arange(0, FINAL_TIMESTEPS * DT, DT)



GQME_TYPE = "SingleState"
# type of reduced (or full) GQME.
# options: Full, PopulationsOnly, SingleState, SubsetStates
STATES = ["11"]
# state(s) to be looking at for SingleState or SubsetStates. It isn't
# necessary to set this for Full or PopulationsOnly because the code is designed
# to create the right arrays for those automatically.
INITIAL_STATE = "00" # initial state

### setting the number of states and array of states strings based on the
### GQME_TYPE
states = [] # array with states in the subset of interest
initInSubset = False
if GQME_TYPE == "Full":
    numStates = DOF_E_SQ # number of states in subset
    for i in range(DOF_E):
        for j in range(DOF_E):
            statesStr = "%s%s"%(i,j)
            states.append(statesStr)

    initInSubset = True
    initialIndex = DOF_E * int(INITIAL_STATE[0]) + int(INITIAL_STATE[1])

elif GQME_TYPE == "PopulationsOnly":
    numStates = DOF_E
    for i in range(DOF_E):
        statesStr = "%s%s"%(i,i)
        states.append(statesStr)

    initInSubset = True
    initialIndex = int(INITIAL_STATE[0])

elif GQME_TYPE == "SubsetStates":
    numStates = len(STATES)
    states = STATES

    if INITIAL_STATE in states:
        initInSubset = True
        initialIndex = states.index(INITIAL_STATE)
elif GQME_TYPE == "SingleState":
    numStates = len(STATES)
    states = STATES
    if numStates != 1:
        print("ERROR: More than one state in STATES with GQME_TYPE = SingleState")

    if states[0] == INITIAL_STATE:
        initInSubset = True
        initialIndex = 0
else:
    print("ERROR: GQME_TYPE not Full, PopulationsOnly, SubsetStates, or SingleState.")



def printTime(outputTime):
    if outputTime / (60. * 60.) > 2:
        return "%.3f hours"%(outputTime / (60. * 60.))
    elif outputTime / 60. > 2:
        return "%.3f minutes"%(outputTime / 60.)
    else:
        return "%.3f seconds"%outputTime

def graph4x4(real_imag, modelNum, legendPos_x, legendPos_y, quantityStr,
             graphStr, timeDict, quantityDict):

    # specifying linestyles, widths, and colors
    linestyles = ["-", "--", "-.", ":", ":", ":", "-.", ":"]
    colors = ['b', 'r', 'm', 'g', 'c', 'y']
    linewidths = [2,2,2,2,3,3]

    fig = plt.figure(figsize = (18,10.5))

    # creates the 4x4 graphs
    ax = []
    for i in range(0,4):
        for j in range(0,4):
            ax.append(plt.subplot2grid((4, 4), (i, j)))

    # sets the spacing between plots
    plt.subplots_adjust(wspace = 0.85, hspace = 0)

    # pulling the time and quantity from the dictionaries
    time = timeDict[graphStr]
    quantity = quantityDict[graphStr]

    # look at real or imag part depending on real_imag input
    if real_imag == "Real":
        quantity = quantity.real
    elif real_imag == "Imag":
        quantity = quantity.imag
    else:
        print("ERROR: real_imag value not Real or Imag")
        return

    # making sure the time values cut off at the limit of the quantity
    # so that their lengths match for the plot
    rangeLimit = len(quantity[:,0,0])

    # loops to plot graphs
    for j in range(0, 4):
        # ab indices of quantity_{abcd}
        l = str(int(j/DOF_E)) + str(int(j%DOF_E))

        # plotting the quantity
        for i in range(4):
            ax[i + j*4].plot(time[0:rangeLimit], quantity[:,j,i],
                             color = colors[0], linewidth = linewidths[0],
                             linestyle = linestyles[0], label = r'%s'%graphStr)

        # top 3 rows of graphs
        if j < 3:
            for i in range(4):
                # since the graphs share x-axes, we need to turn off the ticks
                # for the upper three graphs in each column
                ax[i + j*4].set(xticks=[])

                # makes the y tick values larger
                ax[i + j*4].tick_params(axis='y', labelsize=16)

                # controls the number of y ticks
                ax[i + j*4].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

        # bottom row of graphs
        else:
            for i in range(4):
                # makes both tick values larger
                ax[i + j*4].tick_params(axis='both', labelsize=16)

                # controls the number of ticks
                ax[i + j*4].yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax[i + j*4].xaxis.set_major_locator(MaxNLocator(nbins=5))

        # y labels for U and F
        if len(quantityStr) == 1:
            ax[0 + j*4].set_ylabel(r'${\cal %s}_{%s00}$'%(quantityStr, l),fontsize = 28)
            ax[1 + j*4].set_ylabel(r'${\cal %s}_{%s01}$'%(quantityStr, l),fontsize = 28)
            ax[2 + j*4].set_ylabel(r'${\cal %s}_{%s10}$'%(quantityStr, l),fontsize = 28)
            ax[3 + j*4].set_ylabel(r'${\cal %s}_{%s11}$'%(quantityStr, l),fontsize = 28)

        # y labels for Fdot
        elif quantityStr == "Fdot":
            ax[0 + j*4].set_ylabel(r'$\dot{\cal F}_{%s00}$'%(l),fontsize = 28)
            ax[1 + j*4].set_ylabel(r'$\dot{\cal F}_{%s01}$'%(l),fontsize = 28)
            ax[2 + j*4].set_ylabel(r'$\dot{\cal F}_{%s10}$'%(l),fontsize = 28)
            ax[3 + j*4].set_ylabel(r'$\dot{\cal F}_{%s11}$'%(l),fontsize = 28)

    # sets x labels on bottom row of graphs
    for i in range(4):
        ax[15 - i].set_xlabel(r'$\Gamma\tau$',fontsize = 32)

    # puts a buffer on the left side, as y labels have been cut off before
    plt.gcf().subplots_adjust(left=0.15)

    # gets the labels from the first graph
    handles,labels = ax[0].get_legend_handles_labels()
    legend = ax[0].legend(handles,labels,loc = 'upper right',
                          bbox_to_anchor=(legendPos_x * 4.6, legendPos_y * 1.5),
                          fontsize = 20, borderpad=0.2, borderaxespad=0.2,
                          ncol = 1,
                          title = "%s, Model %s"%(real_imag, modelNum))
    # adjusts title settings
    plt.setp(legend.get_title(),fontsize = 20, fontweight='bold')

    # generates a string to differentiate the files
    outputStr = "_%s_"%(real_imag)

    # saves the figure
    plt.savefig("Figures/" + quantityStr + "_Graphs/" + quantityStr
                + "_model%s"%(modelNum) + outputStr + ".pdf", dpi=plt.gcf().dpi,
                bbox_inches='tight')


def printFFdot(abcdStr, timeFFdot, Freal, Fimag, Fdotreal, Fdotimag):

    # outfileStr for F
    outfileFStr = "ProjFree_Output/F_" + abcdStr + PARAM_STR + ".txt"

    # opens and writes F values to file
    f = open(outfileFStr, "w")
    for i in range(len(Freal)):
        f.write("%s\t%s\t%s\n"%(timeFFdot[i], Freal[i], Fimag[i]))
    f.close()

    # outfileStr for Fdot
    outfileFdotStr = "ProjFree_Output/Fdot_" + abcdStr + PARAM_STR + ".txt"

    # opens and writes Fdot values to file
    f = open(outfileFdotStr, "w")
    for i in range(len(Fdotreal)):
        f.write("%s\t%s\t%s\n"%(timeFFdot[i], Fdotreal[i], Fdotimag[i]))
    f.close()

def buildFFdotDict(FFdotStr, dt, timeSteps, timeDict, quantityDict):

    quantity = np.zeros((timeSteps, DOF_E_SQ, DOF_E_SQ), dtype=np.complex_)
    time = np.zeros((timeSteps))
    for j in range(DOF_E_SQ):
        a = str(int(j/DOF_E))
        b = str(int(j%DOF_E))
        for k in range(DOF_E_SQ):
            c = str(int(k/DOF_E))
            d = str(int(k%DOF_E))

            t, quantityReal, quantityImag = np.hsplit(
                np.loadtxt("ProjFree_Output/" + FFdotStr
                           + "_%s%s%s%s"%(a,b,c,d) + PARAM_STR + ".txt"), 3)

            for i in range(timeSteps):
                time[i] = t[i]
                quantity[i][j][k] = quantityReal[i] + 1.j * quantityImag[i]

    timeDict.update({"%.8f ∆t"%dt : time})
    quantityDict.update({"%.8f ∆t"%dt : quantity})

def PrintKernel(numStates, timeVec, kernel):
    global GQME_TYPE

    for j in range(numStates):
        for k in range(numStates):
            statesStr = states[j] + states[k]

            # outfileStr for K
            outfileKStr = "K_Output/" + GQME_TYPE + "/K_" + statesStr
            # adds subset to filename if type is SubsetStates
            if GQME_TYPE == "SubsetStates":
                outfileKStr += "_Subset"
                for l in range(numStates):
                    outfileKStr += "_" + states[l]
            outfileKStr += PARAM_STR + ".txt"

            # opens and writes to file
            f = open(outfileKStr, "w")
            for i in range(len(timeVec)):
                f.write("%s\t%s\t%s\n"%(timeVec[i], kernel[i][j][k].real, kernel[i][j][k].imag))
            f.close()

def graph_K(real_imag, GQMETypes, statesPerType, maxSet, numCols, legendPos_x,
            legendPos_y, time, KDict):
    global MODEL_NUM, DT, DOF_E, PARAM_STR

    # maximum number of states considered, from 1 to 4
    maxStates = len(maxSet)

    # setting colors, linewidths, and linestyles
    fullColor = 'b'
    fullLineWidth = 2
    fullLineStyle = '-'

    popOnlyColor = 'r'
    popOnlyLineWidth = 2
    popOnlyLineStyle = '-'

    singleColor = 'g'
    singleLineWidth = 2
    singleLineStyle = '-'

    # if more than 6 subsets, need to add more colors, widths, and styles
    subsetColor = ['m', 'c', 'y', (0,0,0.5), (0.5,0,0), (0,0.5,0)]
    subsetLineWidth = [2, 2, 2, 2, 2, 2]
    subsetLineStyle = ['--', '--', '--', '--', '--', '--']

    # generates a string to differentiate the files
    outputStr = "_%s"%(real_imag)

    fig = plt.figure(figsize = (19,10))

    # creates the maxStates x maxStates graphs
    ax = []
    for i in range(maxStates):
        for j in range(maxStates):
            ax.append(plt.subplot2grid((maxStates, maxStates), (i, j)))

    # sets the spacing between plots, slightly different b/w real and imag
    if real_imag == "Real":
        plt.subplots_adjust(wspace = 0.75, hspace = 0)
    else:
        plt.subplots_adjust(wspace = 0.73, hspace = 0)

    # making sure the time values cut off at the limit of the quantity
    # so that their lengths match for the plot
    rangeLimit = len(time)

    # because of gradients used to get F and Fdot, end of kernel can be wacky
    frontCut = 2

    # initialize handles for legend
    handles = []

    # count of single states, to ensure only one put in legend
    singleCount = 0

    # count of subsets, for pulling in line colors, widths, and style
    subsetCount = 0

    for indexGQME in range(len(GQMETypes)):
        # pulls in type of GQME
        typeGQME = GQMETypes[indexGQME]

        # plots full memory kernel
        if typeGQME == "Full":
            # pulling in K from dictionary
            full = KDict[typeGQME]
            if real_imag == "Real":
                full = full.real
            elif real_imag == "Imag":
                full = full.imag
            else:
                print("ERROR: real_imag value not Real or Imag")
                return

            # loops to plot graphs
            for j in range(maxStates):
                for i in range(maxStates):
                    ax[i + j * maxStates].plot(time[frontCut:],
                                     full[frontCut:,j,i], color = fullColor,
                                     linewidth = fullLineWidth,
                                     linestyle = fullLineStyle)

            legendLine = Line2D([0], [0], label = "Full", color = fullColor,
                                linewidth = fullLineWidth,
                                linestyle = fullLineStyle)

            handles.extend([legendLine])

            outputStr += "_Full"

        # plots populations-only memory kernel
        elif typeGQME == "PopulationsOnly":
            # pulling in K from dictionary
            popOnly = KDict[typeGQME]
            if real_imag == "Real":
                popOnly = popOnly.real
            elif real_imag == "Imag":
                popOnly = popOnly.imag
            else:
                print("ERROR: real_imag value not Real or Imag")
                return

            ax[0].plot(time[frontCut:], popOnly[frontCut:,0,0],
                       color = popOnlyColor, linewidth = popOnlyLineWidth,
                       linestyle = popOnlyLineStyle)
            ax[maxStates - 1].plot(time[frontCut:], popOnly[frontCut:,0,1],
                       color = popOnlyColor, linewidth = popOnlyLineWidth,
                       linestyle = popOnlyLineStyle)
            ax[maxStates**2 - maxStates].plot(time[frontCut:], popOnly[frontCut:,1,0],
                        color = popOnlyColor, linewidth = popOnlyLineWidth,
                        linestyle = popOnlyLineStyle)
            ax[maxStates**2 - 1].plot(time[frontCut:], popOnly[frontCut:,1,1],
                        color = popOnlyColor, linewidth = popOnlyLineWidth,
                        linestyle = popOnlyLineStyle)

            legendLine = Line2D([0], [0], label = "PopOnly",
                                color = popOnlyColor,
                                linewidth = popOnlyLineWidth,
                                linestyle = popOnlyLineStyle)

            handles.extend([legendLine])

            outputStr += "_PopOnly"

        # plots a single-state memory kernel, based on state in statesPerType
        elif typeGQME == "SingleState":
            # pulling in state string
            state = statesPerType[indexGQME][0]

            # pulling in K from dictionary
            single = KDict["single_%s"%state]
            if real_imag == "Real":
                single = single.real
            elif real_imag == "Imag":
                single = single.imag
            else:
                print("ERROR: real_imag value not Real or Imag")
                return

            # determining indices of state
            index = maxSet.index(state)
            # j = int(state[0])
            # k = int(state[1])

            ### WRONG
            # determines index of the graph based on state
            #graphIndex = DOF_E_SQ * (DOF_E * j + k) + DOF_E * j + k
            graphIndex = maxStates * index + index

            ax[graphIndex].plot(time[frontCut:], single[frontCut:],
                   color = singleColor, linewidth = singleLineWidth,
                   linestyle = singleLineStyle)

            if singleCount == 0:
                legendLine = Line2D([0], [0], label = "Single",
                                    color = singleColor,
                                    linewidth = singleLineWidth,
                                    linestyle = singleLineStyle)

                handles.extend([legendLine])

            outputStr += "_Single_%s"%state

            singleCount += 1

        # plots a subset-states memory kernel, based on states in statesPerType
        elif typeGQME == "SubsetStates":
            # pulling in state string
            states = statesPerType[indexGQME]

            #print("states %s, subsetCount %s"%(states, subsetCount))

            numStates = len(states)

            subsetStr = "Subset"
            subsetList = states[0]
            for l in range(numStates):
                subsetStr += "_" + states[l]
                if l != 0:
                    subsetList += ", " + states[l]

            # pulling in K from dictionary
            subset = KDict[subsetStr]
            if real_imag == "Real":
                subset = subset.real
            elif real_imag == "Imag":
                subset = subset.imag
            else:
                print("ERROR: real_imag value not Real or Imag")
                return

            for j in range(numStates):
                # determining indices of state j
                index_j = maxSet.index(states[j])
                # a = int(states[j][0])
                # b = int(states[j][1])
                for k in range(numStates):
                    # determining indices of state k
                    index_k = maxSet.index(states[k])
                    # c = int(states[k][0])
                    # d = int(states[k][1])

                    # determines index of the graph based on state
                    graphIndex = maxStates * index_j + index_k
                    #graphIndex = DOF_E_SQ * (DOF_E * a + b) + DOF_E * c + d

                    ax[graphIndex].plot(time[frontCut:], subset[frontCut:,j,k],
                                        color = subsetColor[subsetCount],
                                        linewidth = subsetLineWidth[subsetCount],
                                        linestyle = subsetLineStyle[subsetCount])

            legendLine = Line2D([0], [0], label = r'Subset %s'%subsetList,
                                color = subsetColor[subsetCount],
                                linewidth = subsetLineWidth[subsetCount],
                                linestyle = subsetLineStyle[subsetCount])

            handles.extend([legendLine])

            outputStr += "_" + subsetStr

            subsetCount += 1

    # loops to plot graphs
    for j in range(maxStates):
        # ab indices of quantity_{abcd}
        #l_array = ["DD", "DA", "AD", "AA"]
        #l = l_array[j]
        l = maxSet[j]

        # all rows of graphs besides the bottom row
        if j < (maxStates - 1):
            for i in range(maxStates):
                # since the graphs share x-axes, we need to turn off the ticks
                # for the upper three graphs in each column
                ax[i + j * maxStates].set(xticks=[])

                # makes the y tick values larger
                ax[i + j * maxStates].tick_params(axis='y', labelsize=16)

                # controls the number of y ticks
                ax[i + j * maxStates].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

        # bottom row of graphs
        else:
            for i in range(maxStates):
                # makes both tick values larger
                ax[i + j * maxStates].tick_params(axis='both', labelsize=16)

                # controls the number of ticks
                ax[i + j * maxStates].yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax[i + j * maxStates].xaxis.set_major_locator(MaxNLocator(nbins=5))

        # y labels
        for i in range(maxStates):
            k = maxSet[i]
            ax[i + j * maxStates].set_ylabel(r'${\cal K}_{%s%s}$'%(l, k),fontsize = 28)

        for i in range(maxStates):
            # sets limit of x-axis
            ax[i + j * maxStates].set_xlim(right = time[-1])

    # sets x labels on bottom row of graphs
    for i in range(maxStates):
        ax[maxStates**2 - i - 1].set_xlabel(r'$\Gamma\tau$',fontsize = 32)

    # puts a buffer on the left side, as y labels have been cut off before
    plt.gcf().subplots_adjust(left=0.15)

    legend = ax[0].legend(handles=handles, loc = 'upper right',
                          bbox_to_anchor=(legendPos_x[maxStates - 1] * 5.75,
                                          legendPos_y[maxStates - 1] * 2.),
                          fontsize = 20, borderpad=0.2, borderaxespad=0.2,
                          ncol = numCols,
                          title = "%s Part of the Memory Kernels "%(real_imag)
                          + "for Model %s and %.8f ∆t"%(MODEL_NUM, DT))
    # adjusts title settings
    plt.setp(legend.get_title(),fontsize = 20, fontweight='bold')

    # saves the figure
    plt.savefig("Figures/K_Graphs/K_model%s"%MODEL_NUM + outputStr + PARAM_STR
                + ".pdf", dpi=plt.gcf().dpi, bbox_inches='tight')


def buildKDict(GQMETypes, statesPerType, KDict):

    if len(GQMETypes) != len(statesPerType):
        print("ERROR: length of GQMETypes not equal to length of statesPerType.")
        print("Exiting.")
        return

    timeSteps_U = TIME_STEPS
    # because of gradients used to get F and Fdot, end of kernel can be wacky
    timeSteps = TIME_STEPS - 2

    time = np.zeros((timeSteps))

    for indexGQME in range(len(GQMETypes)):
        typeGQME = GQMETypes[indexGQME]
        if typeGQME == "Full":
            full = np.zeros((timeSteps, DOF_E_SQ, DOF_E_SQ), dtype=np.complex_)
            # reading in and storing K full
            for j in range(DOF_E_SQ):
                a = str(int(j/DOF_E))
                b = str(int(j%DOF_E))
                for k in range(DOF_E_SQ):
                    c = str(int(k/DOF_E))
                    d = str(int(k%DOF_E))

                    t, fullReal, fullImag = np.hsplit(
                        np.loadtxt("K_Output/Full/K_%s%s%s%s"%(a,b,c,d)
                        + PARAM_STR + ".txt"), 3)

                    for i in range(timeSteps):
                        time[i] = t[i]
                        full[i][j][k] = fullReal[i] + 1.j * fullImag[i]

            KDict.update({typeGQME : full})
        elif typeGQME == "PopulationsOnly":
            popOnly = np.zeros((timeSteps, DOF_E, DOF_E), dtype=np.complex_)
            # reading in and storing K pop-only
            for j in range(DOF_E):
                ab = str(j) + str(j)
                for k in range(DOF_E):
                    cd = str(k) + str(k)

                    t, popOnlyReal, popOnlyImag = np.hsplit(
                        np.loadtxt("K_Output/PopulationsOnly/K_%s%s"%(ab,cd)
                        + PARAM_STR + ".txt"), 3)

                    for i in range(timeSteps):
                        time[i] = t[i]
                        popOnly[i][j][k] = popOnlyReal[i] + 1.j * popOnlyImag[i]

            KDict.update({typeGQME : popOnly})
        elif typeGQME == "SingleState":
            single = np.zeros((timeSteps), dtype=np.complex_)

            # reading in and storing K single state
            j = statesPerType[indexGQME][0]
            abcd = j + j

            t, singleReal, singleImag = np.hsplit(
                np.loadtxt("K_Output/SingleState/K_" + abcd + PARAM_STR + ".txt"), 3)

            for i in range(timeSteps):
                time[i] = t[i]
                single[i] = singleReal[i] + 1.j * singleImag[i]

            KDict.update({"single_%s"%j : single})
        elif typeGQME == "SubsetStates":
            states = statesPerType[indexGQME]
            numStates = len(states)
            subset = np.zeros((timeSteps, numStates, numStates), dtype=np.complex_)

            subsetStr = "Subset"
            for l in range(numStates):
                subsetStr += "_" + states[l]

            # reading in and storing K subset states
            for j in range(numStates):
                for k in range(numStates):
                    abcd = states[j] + states[k]

                    t, subsetReal, subsetImag = np.hsplit(
                        np.loadtxt("K_Output/SubsetStates/K_" + abcd + "_"
                                   + subsetStr + PARAM_STR + ".txt"), 3)

                    for i in range(timeSteps):
                        time[i] = t[i]
                        subset[i][j][k] = subsetReal[i] + 1.j * subsetImag[i]

            KDict.update({subsetStr : subset})

    return time

def PrintITerm(numStates, timeVec, iTerm):
    global GQME_TYPE

    for j in range(numStates):
        statesStr = states[j] + "_startingIn_" + INITIAL_STATE

        # outfileStr for K
        outfileIStr = "I_Output/" + GQME_TYPE + "/I_" + statesStr
        if GQME_TYPE == "SubsetStates":
            outfileIStr += "_Subset"
            for l in range(numStates):
                outfileIStr += "_" + states[l]
        outfileIStr += PARAM_STR + ".txt"

        f = open(outfileIStr, "w")

        for i in range(len(timeVec)):
            f.write("%s\t%s\t%s\n"%(timeVec[i], iTerm[i][j].real, iTerm[i][j].imag))
        f.close()

def graph_I(real_imag, GQMETypes, statesPerType, maxSet, numCols, legendPos_x,
            legendPos_y, time, IDict):
    global MODEL_NUM, DT, DOF_E, PARAM_STR

    # maximum number of states considered, from 1 to 3
    maxStates = len(maxSet)

    # setting colors, linewidths, and linestyles
    singleColor = 'g'
    singleLineWidth = 2
    singleLineStyle = '-'

    # if more than 6 subsets, need to add more colors, widths, and styles
    subsetColor = ['m', 'c', 'y', (0,0,0.5), (0.5,0,0), (0,0.5,0)]
    subsetLineWidth = [2, 2, 2, 2, 2, 2]
    subsetLineStyle = ['--', '--', '--', '--', '--', '--']

    # generates a string to differentiate the files
    outputStr = "_%s"%(real_imag)

    fig = plt.figure(figsize = (10, 8  * maxStates))

    # creates the maxStates graphs
    ax = []
    for i in range(maxStates):
        ax.append(plt.subplot2grid((maxStates, 1), (i, 0)))

    # sets the spacing between plots
    plt.subplots_adjust(hspace = 0)

    # making sure the time values cut off at the limit of the quantity
    # so that their lengths match for the plot
    rangeLimit = len(time)

    # because of gradients used to get F and Fdot, end of kernel can be wacky
    frontCut = 2

    # initialize handles for legend
    handles = []

    # count of single states, to ensure only one put in legend
    singleCount = 0

    # count of subsets, for pulling in line colors, widths, and style
    subsetCount = 0

    for indexGQME in range(len(GQMETypes)):
        # pulls in type of GQME
        typeGQME = GQMETypes[indexGQME]

        # plots a single-state memory kernel, based on state in statesPerType
        if typeGQME == "SingleState":
            # pulling in state string
            state = statesPerType[indexGQME][0]

            # pulling in K from dictionary
            single = IDict["single_%s"%state]
            if real_imag == "Real":
                single = single.real
            elif real_imag == "Imag":
                single = single.imag
            else:
                print("ERROR: real_imag value not Real or Imag")
                return

            # determining indices of state
            index = maxSet.index(state)

            ax[index].plot(time[frontCut:], single[frontCut:],
                           color = singleColor, linewidth = singleLineWidth,
                           linestyle = singleLineStyle)

            if singleCount == 0:
                legendLine = Line2D([0], [0], label = "Single",
                                    color = singleColor,
                                    linewidth = singleLineWidth,
                                    linestyle = singleLineStyle)

                handles.extend([legendLine])

            outputStr += "_Single_%s"%state

            singleCount += 1

        # plots a subset-states memory kernel, based on states in statesPerType
        elif typeGQME == "SubsetStates":
            # pulling in state string
            states = statesPerType[indexGQME]

            numStates = len(states)

            subsetStr = "Subset"
            subsetList = states[0]
            for l in range(numStates):
                subsetStr += "_" + states[l]
                if l != 0:
                    subsetList += ", " + states[l]

            # pulling in K from dictionary
            subset = IDict[subsetStr]
            if real_imag == "Real":
                subset = subset.real
            elif real_imag == "Imag":
                subset = subset.imag
            else:
                print("ERROR: real_imag value not Real or Imag")
                return

            for j in range(numStates):
                # determining indices of state j
                index_j = maxSet.index(states[j])

                ax[index_j].plot(time[frontCut:], subset[frontCut:,j],
                               color = subsetColor[subsetCount],
                               linewidth = subsetLineWidth[subsetCount],
                               linestyle = subsetLineStyle[subsetCount])

            legendLine = Line2D([0], [0], label = r'Subset %s'%subsetList,
                                color = subsetColor[subsetCount],
                                linewidth = subsetLineWidth[subsetCount],
                                linestyle = subsetLineStyle[subsetCount])

            handles.extend([legendLine])

            outputStr += "_" + subsetStr

            subsetCount += 1

    # loops to plot graphs
    for j in range(maxStates):
        # ab indices of I_{ab}
        l = maxSet[j]

        # top 3 rows of graphs
        if j < (maxStates - 1):
            # since the graphs share x-axes, we need to turn off the ticks
            # for the upper three graphs in each column
            ax[j].set(xticks=[])

            # makes the y tick values larger
            ax[j].tick_params(axis='y', labelsize=16)

            # controls the number of y ticks
            ax[j].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

        # bottom row of graphs
        else:
            # makes both tick values larger
            ax[j].tick_params(axis='both', labelsize=16)

            # controls the number of ticks
            ax[j].yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax[j].xaxis.set_major_locator(MaxNLocator(nbins=5))

            # sets x label on bottom graph
            ax[j].set_xlabel(r'$\Gamma t$',fontsize = 32)

        # y labels
        ax[j].set_ylabel(r'$\hat{I}_{%s}(t)$'%(l),fontsize = 28)

        # sets limit of x-axis
        ax[j].set_xlim(right = time[-1])

    # puts a buffer on the left side, as y labels have been cut off before
    plt.gcf().subplots_adjust(left=0.15)

    legend = ax[0].legend(handles=handles, loc = 'upper right',
                          bbox_to_anchor=(legendPos_x * 5.75, legendPos_y * 2.),
                          fontsize = 20, borderpad=0.2, borderaxespad=0.2,
                          ncol = numCols,
                          title = "%s Part of the "%(real_imag)
                          + "Inhomogeneous Terms\n"
                          + "  for Model %s and %.8f ∆t"%(MODEL_NUM, DT))
    # adjusts title settings
    plt.setp(legend.get_title(),fontsize = 20, fontweight='bold')

    # saves the figure
    plt.savefig("Figures/I_Graphs/I_model%s"%MODEL_NUM + outputStr + PARAM_STR
                + ".pdf", dpi=plt.gcf().dpi, bbox_inches='tight')

def buildIDict(GQMETypes, statesPerType, IDict):
    global PARAM_STR, TIME_STEPS, DT

    if len(GQMETypes) != len(statesPerType):
        print("ERROR: length of GQMETypes not equal to length of statesPerType.")
        print("Exiting.")
        return

    timeSteps_U = TIME_STEPS
    # because of gradients used to get F and Fdot, end of kernel can be wacky
    timeSteps = TIME_STEPS - 2

    time = np.zeros((timeSteps))

    for indexGQME in range(len(GQMETypes)):
        typeGQME = GQMETypes[indexGQME]
        if typeGQME == "SingleState":
            single = np.zeros((timeSteps), dtype=np.complex_)

            # reading in and storing K single state
            j = statesPerType[indexGQME][0]
            abcd = j + "_startingIn_" + INITIAL_STATE

            t, singleReal, singleImag = np.hsplit(
                np.loadtxt("I_Output/SingleState/I_" + abcd + PARAM_STR + ".txt"), 3)

            for i in range(timeSteps):
                time[i] = t[i]
                single[i] = singleReal[i] + 1.j * singleImag[i]

            IDict.update({"single_%s"%j : single})
        elif typeGQME == "SubsetStates":
            states = statesPerType[indexGQME]
            numStates = len(states)
            subset = np.zeros((timeSteps, numStates), dtype=np.complex_)

            subsetStr = "Subset"
            for l in range(numStates):
                subsetStr += "_" + states[l]

            # reading in and storing K subset states
            for j in range(numStates):
                abcd = states[j] + "_startingIn_" + INITIAL_STATE

                t, subsetReal, subsetImag = np.hsplit(
                    np.loadtxt("I_Output/SubsetStates/I_" + abcd + "_"
                               + subsetStr + PARAM_STR + ".txt"), 3)

                for i in range(timeSteps):
                    time[i] = t[i]
                    subset[i][j] = subsetReal[i] + 1.j * subsetImag[i]

            IDict.update({subsetStr : subset})

    return time

def PrintGQMESigma(numStates, states, timeVec, sigma):
    global GQME_TYPE, INITIAL_STATE, PARAM_STRING_GQME

    for j in range(numStates):
        statesStr = states[j] + "_startingIn_" + INITIAL_STATE

        # outfileStr for K
        outfileGQMEStr = "GQME_Output/" + GQME_TYPE + "/Sigma_" + statesStr
        if GQME_TYPE == "SubsetStates":
            outfileGQMEStr += "_Subset"
            for l in range(numStates):
                outfileGQMEStr += "_" + states[l]
        outfileGQMEStr += PARAM_STR_GQME + ".txt"

        f = open(outfileGQMEStr, "w")

        for i in range(len(timeVec)):
            f.write("%s\t%s\t%s\n"%(timeVec[i], sigma[i][j].real, sigma[i][j].imag))
        f.close()

def graphSigmaZ(GQMETypes, numCols, legendPos_x, legendPos_y, time_U, U, time, sigmaZDict):
    global MODEL_NUM, PARAM_STR_GQME

    color_full = 'b'
    color_pop = 'r'
    color_single = 'g'
    # if more than 6 subsets, more colors will need to be added
    color_subsets = ['m', 'c', 'y', (0,0,0.5), (0.5,0,0), (0,0.5,0)]
    graphxlim = time_U[-1]

    fig = plt.figure(figsize = (8,6))
    ax = fig.add_axes([0.05,0.05,0.9,0.9])

    # calling in and plotting QUAPI results
    t_exact, sigz_exact = np.hsplit(
        np.loadtxt("QUAPI_results/Kelly-sigz-xi%swc%s-QUAPI"%(XI, OMEGA_C)
        + ".txt"),2)
    plt.plot(t_exact, sigz_exact, 'k.', label='QuAPI',linewidth = 3,
             markersize = 20)

    #plotting U
    plt.plot(time_U, U, '-', color = (0.5, 0.5, 0.5), label = "TT-TFD", linewidth = 4)

    GQMECount = 0
    subsetCount = 0
    outputStr = ""
    # plotting the population difference
    for key in sigmaZDict:
        if key[:3] == "Ful":
            plt.plot(time, sigmaZDict[key], '--', color = color_full,
                     label='Full', linewidth = 9)
            GQMECount += 1
        elif key[:3] == "Pop":
            plt.plot(time, sigmaZDict[key], '-.', color = color_pop,
                     label='Pop-Only', linewidth = 9)
            GQMECount += 1
        elif key[:3] == "Sin":
            plt.plot(time, sigmaZDict[key], ':', color = color_single,
                     label='Single', linewidth = 9)
            GQMECount += 1
        else:
            labelPrefix = "Subset "
            labelSubset = GQMETypes[GQMECount][0][7:9] + ", " + GQMETypes[GQMECount][0][10:12]
            if len(GQMETypes[GQMECount][0]) > 12:
                labelSubset += ", " + GQMETypes[GQMECount][0][13:15]
            if len(GQMETypes[GQMECount]) == 2:
                labelPrefix = "Subsets "
                labelSubset += "\n& " + GQMETypes[GQMECount][1][7:9] + ", "
                labelSubset += GQMETypes[GQMECount][1][10:12]
                if len(GQMETypes[GQMECount][1]) > 12:
                    labelSubset += ", " + GQMETypes[GQMECount][1][13:15]

            lineStyles = ["--", "-.", ":"]
            plt.plot(time, sigmaZDict[key], linestyle = lineStyles[subsetCount],
                     color = color_subsets[subsetCount],
                     label = labelPrefix + labelSubset, linewidth = 5)
            GQMECount += 1
            subsetCount += 1

        outputStr += "_[" + key + "]"

    plt.xlabel('$\Gamma\, t$', fontsize = 32)
    plt.ylabel('$\sigma_z(t)$',fontsize = 36)
    #plt.ylabel('$\sigma_{00}(t)$',fontsize = 36)

    y1 = 1.
    if XI == 0.1 and OMEGA_C == 1 and EPSILON == 1:
        #Model 1: xi = 0.1 wc = 1:
        y0=-0.05
    elif XI == 0.1 and OMEGA_C == 2:
        #Model 2: xi = 0.1 wc = 2
        y0=-0.45
    elif OMEGA_C == 7.5:
        #Model 3: xi = 0.1 wc = 7.5
        y0=-1.0
    elif XI == 0.4:
        #Model 4: xi = 0.4 wc = 2
        y0=-1.
    elif EPSILON == 0 and OMEGA_C == 1:
        #Model 5: xi = 0.1 wc = 1  e = 0
        y0=-0.75
    else:
        #Model 6: xi = 0.2 wc = 2.5  e = 0
        y0=-0.75
        y1 = 1.

    bounds = [0, graphxlim, y0, y1]
    plt.axis(bounds)
    plt.tick_params(axis = 'both', labelsize = 18)

    legend = plt.legend(loc = 'upper right',
                        bbox_to_anchor=(legendPos_x * 5.75, legendPos_y * 2.),
                        fontsize = 20, borderpad=0.2, borderaxespad=0.2,
                        ncol = numCols,
                        title = '$\sigma_z(t)$ for Model %s and %.8f ∆t'%(MODEL_NUM, DT))
    # adjusts title settings
    plt.setp(legend.get_title(),fontsize = 20, fontweight='bold')

    #plt.legend(loc='upper right',fontsize = 22, ncol = 2, borderpad=0.2,
    #           borderaxespad=0.2, columnspacing=2, frameon=False)
    #plt.suptitle(r'$\sigma_z(t) = \sigma_{00}(t) - \sigma_{11}(t)$', y = 0.99, fontsize = 36)
    #plt.suptitle(r'$\sigma_z(t)$ for Model %s'%MODEL_NUM , y = 1.01, x=.48,
    #             fontsize = 28, fontweight='bold')
    plt.savefig("Figures/GQME_Graphs/Sig_Z_model%s"%MODEL_NUM + outputStr
                + PARAM_STR_GQME + ".pdf", dpi=plt.gcf().dpi,
                bbox_inches='tight')

def dataUandGQME(GQMETypes):
    global PARAM_STR, PARAM_STR_GQME, MODEL_NUM, TIME_STEPS

    sigmaZDict = {}

    # reading in U
    time_U, UReal_00, UImag_00 = np.hsplit(
        np.loadtxt("U_Output/U_0000" + PARAM_STR + ".txt"),3)
    time_U, UReal_11, UImag_11 = np.hsplit(
        np.loadtxt("U_Output/U_1100" + PARAM_STR + ".txt"),3)

    # calculating sigma_z for U
    U = np.zeros((len(time_U)))
    for i in range(len(time_U)):
        U[i] = UReal_00[i] - UReal_11[i]

    for indexGQME in range(len(GQMETypes)):
        arrayGQME = GQMETypes[indexGQME]
        numGQME = len(arrayGQME)

        # reading in sigma_00
        if arrayGQME[0][:3] == "Sub":
            time, real00, imag00 = np.hsplit(
                np.loadtxt("GQME_Output/SubsetStates/Sigma_00_startingIn_00_"
                           + arrayGQME[0] + PARAM_STR_GQME + ".txt"), 3)
        else:
            time, real00, imag00 = np.hsplit(
                np.loadtxt("GQME_Output/" + arrayGQME[0]
                           + "/Sigma_00_startingIn_00" + PARAM_STR_GQME
                           + ".txt"), 3)

        # reading in sigma_11
        if arrayGQME[numGQME - 1][:3] == "Sub":
            time, real11, imag11 = np.hsplit(
                np.loadtxt("GQME_Output/SubsetStates/Sigma_11_startingIn_00_"
                           + arrayGQME[numGQME - 1] + PARAM_STR_GQME + ".txt"), 3)
        else:
            time, real11, imag11 = np.hsplit(
                np.loadtxt("GQME_Output/" + arrayGQME[numGQME - 1]
                           + "/Sigma_11_startingIn_00" + PARAM_STR_GQME
                           + ".txt"), 3)

        dictKey = arrayGQME[0]
        if numGQME == 2:
            dictKey += "_" + arrayGQME[1]

        sigma_z = real00 - real11

        sigmaZDict.update({dictKey : sigma_z})

    return time_U, U, time, sigmaZDict




