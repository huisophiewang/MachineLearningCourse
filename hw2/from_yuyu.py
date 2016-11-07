import numpy as np
import math

nnInput = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]
nnOutput = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]]

"""
input weights doesn't contain bias weight
This only contains data, no connection information
"""
class NNNode:
    inputweights = None; input = []; bias = 0;
    output = 0; error = 0; predict = 0;
    position = ""; index = -1;

    def __init__(self, i = 0):
        self.index = i;

    def SetPosition(self, pstr):
        self.position = pstr;

    def ComputeInputOutput(self, lowerLayer):
        # init weight
        if (lowerLayer is not None and self.inputweights is None):
            self.inputweights = np.random.random(len(lowerLayer));
            #self.inputweights = np.zeros(len(lowerLayer));
        # set input
        if (self.position == "input"):
            self.predict = self.input[0];
        else:
            self.input = [];
            for pn in lowerLayer:
                self.input.append(pn.GetPredict());
            I = sum([self.inputweights[x] * self.input[x] for x in range(0, len(self.inputweights))]) + self.bias;
            self.predict = 1 / (1 + math.e**(-I));
        return self.input, self.predict;

    def ComputeError(self, higherLayer):
        if (self.position == "output"):
            self.error = self.predict * (1 - self.predict) * (self.output - self.predict);
        else:
            errorVec = [higherLayer[i].GetError() * higherLayer[i].GetWeights()[self.index] for i in range(0, len(higherLayer))];
            self.error = sum(errorVec) * self.predict * (1 - self.predict);
        return self.error;

    def UpdateWeights(self, lowerLayer, alpha):
        if (self.position == "input"): return;
        for i in range(0, len(self.inputweights)):
            self.inputweights[i] = self.inputweights[i] + alpha * self.error * lowerLayer[i].GetPredict();
            self.bias = self.bias + alpha * self.error;
        return;

    def GetInput(self):
        return self.input;

    def GetPredict(self):
        return self.predict;

    def GetOutput(self):
        return self.output;

    def GetError(self):
        return self.error;

    def GetWeights(self):
        return self.inputweights;

    def SetInput(self, inp):
        self.input = inp;

    def SetOutput(self, o):
        self.output = o;

def setup():
    """
    Setup
    """
    NNNodeLayers = [];
    ''' layer 1 '''
    layer1 = [];
    for i in range(0, 8):
        n = NNNode(i); n.SetPosition("input");
        layer1.append(n);
    NNNodeLayers.append(layer1);
    ''' layer 2 '''
    layer2 = [];
    for i in range(0, 3):
        n = NNNode(i);
        layer2.append(n);
    NNNodeLayers.append(layer2);
    ''' layer 3 '''
    layer3 = [];
    for i in range(0, 8):
        n = NNNode(i); n.SetPosition("output");
        layer3.append(n);
    NNNodeLayers.append(layer3);
    return NNNodeLayers

def algorithm(NNNodeLayers):
    """
    Algorithm
    """
    learningRate = 0.2;
    threshold = 0.8;
    noLoops = 0;
    cvg = False;
    noSkips = 0;
    finallist = [];
    while (not cvg):
        for index in range(0, len(nnInput)):
            noLoops = noLoops + 1;
            print 'loop: ' + str(noLoops) + ' data: ' + str(index);
            din = nnInput[index];
            dout = nnOutput[index];
            hidden = [];
            for i in range(0, len(din)):
                NNNodeLayers[0][i].SetInput([din[i]]);
            for i in range(0, len(dout)):
                NNNodeLayers[2][i].SetOutput(dout[i]);
    
            #print '--- forward computation --- ';
            for i in range(0, len(NNNodeLayers)):
                prevLayer = None;
                if (i != 0): prevLayer = NNNodeLayers[i - 1];
                for j in range(0, len(NNNodeLayers[i])):
                    NNNodeLayers[i][j].ComputeInputOutput(prevLayer);
                    print str(i) + ' ' + str(j) + ' input: ' + str(NNNodeLayers[i][j].GetInput()) + ' output: ' + str(NNNodeLayers[i][j].GetPredict());
                    if (i == 1):
                        hidden.append(NNNodeLayers[i][j].GetPredict());
    
            #print '---check---'
            skip = True;
            for i in range(0, len(NNNodeLayers[2])):
                if NNNodeLayers[2][i].GetPredict() >= threshold: result = 1;
                else:   result = 0;
                if (result != NNNodeLayers[2][i].GetOutput()):
                    skip = False;
                    noSkips = 0;
                    finallist = [];
                    break;
            if (skip is True):
                print "Skip is True"
                print 'loop: ' + str(noLoops) + ' data: ' + str(index) + ' No. Skips: ' + str(noSkips);
                noSkips = noSkips + 1;
                finallist = finallist + hidden;
                if (noSkips == len(nnInput)):
                    print 'Total loop: ' + str(noLoops / float(len(nnInput)));
                    for j in range(0, len(NNNodeLayers[2])):
                        print str(2) + ' ' + str(j) + ' final weights: ' + str(NNNodeLayers[2][j].GetWeights());
                    cvg = True;
                    print finallist;
                    break;
                else:
                    continue;
    
            #print '--- backward computation --- '
            for i in range(len(NNNodeLayers) - 1, -1, -1):
                prevLayer = None;
                if (i != len(NNNodeLayers) - 1): prevLayer = NNNodeLayers[i + 1];
                for j in range(0, len(NNNodeLayers[i])):
                    NNNodeLayers[i][j].ComputeError(prevLayer);
                    #print str(i) + ' ' + str(j) + ' error: ' + str(NNNodeLayers[i][j].GetError());
    
            # print '--- update weights --- '
            for i in range(0, len(NNNodeLayers)):
                prevLayer = None;
                if (i != 0): prevLayer = NNNodeLayers[i - 1];
                for j in range(0, len(NNNodeLayers[i])):
                    NNNodeLayers[i][j].UpdateWeights(prevLayer, learningRate);
                    #print str(i) + ' ' + str(j) + ' updateweights: ' + str(NNNodeLayers[i][j].GetWeights());
                    
if __name__ == '__main__':
    NNNodeLayers = setup()
    #print NNNodeLayers
    algorithm(NNNodeLayers)
