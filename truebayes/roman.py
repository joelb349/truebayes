import os
import time
import pkg_resources

import numpy as np
import math

import torch
import torch.optim as optim

import truebayes
from truebayes.network import makenet
from truebayes.utils import numpy2cuda
from truebayes.geometry import qdim, xstops

# load the standard ROMAN network

# layers = [4,8,16,32,64,128] + [256] * 20 + [241]

# alvin = makenet(layers, softmax=False)

# ar, ai = alvin(), alvin()

# datadir = pkg_resources.resource_filename(__name__, 'data/')

# if torch.cuda.is_available():
#   ar.load_state_dict(torch.load(os.path.join(datadir, 'roman/ar-state.pt')))
#   ai.load_state_dict(torch.load(os.path.join(datadir, 'roman/ai-state.pt')))
# else:
#   ar.load_state_dict(torch.load(os.path.join(datadir, 'roman/ar-state.pt'), map_location=torch.device('cpu')))
#   ai.load_state_dict(torch.load(os.path.join(datadir, 'roman/ai-state.pt'), map_location=torch.device('cpu')))

# ar.eval()
# ai.eval()

varx = ['M', 'tc']

## Produce Training Signals and Train Neural Network 


def syntrain(size, region=None, varx=varx, seed=None, varall=False,
                single=True, noise=0):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    with torch.no_grad():
        xs = torch.zeros((size, len(region)), dtype=torch.float, device=device)

        for i, r in enumerate(region):
            xs[:,i] = r[0] + (r[1] - r[0])*torch.rand((size,), dtype=torch.float, device=device)

        xr = xs.detach().cpu().double().numpy()
        
        signal = np.apply_along_axis(sinc_f, 1, xr)
        
        signal_r, signal_i = numpy2cuda(signal), 0
        
        alphas = torch.zeros((size, 500), dtype=torch.float if single else torch.double, device=device)
        
        ##Normalize the signal elements
        normalize = torch.sqrt(torch.sum(signal_r*signal_r + signal_i*signal_i, dim=1))
        
        ##Vary the signal amplitudes
        amp = [8,12]
        const = numpy2cuda(np.random.uniform(*amp,size=size))

        ##Add noise and normalize
        alphas[:,:] = const[:,np.newaxis]*signal_r/normalize[:,np.newaxis] + (noise*torch.randn((size,500), device=device))
        ##alphas[:,1::2] = const[:, np.newaxis]*(signal_i / normalize[:,np.newaxis]) + (noise*torch.randn((size,133), device=device))
    
    del xs, signal_r, signal_i 
    
    for i,r in enumerate(region):
        xr[:,i] = (xr[:,i] - r[0]) / (r[1] - r[0])    
    
    if isinstance(varx, list):
        ix = ['M','tc'].index(varx[0])
        jx = ['M','tc'].index(varx[1])    

        i = np.digitize(xr[:,ix], xstops, False) - 1
        i[i == -1] = 0; i[i == qdim] = qdim - 1
        px = np.zeros((size, qdim), 'd'); px[range(size), i] = 1

        j = np.digitize(xr[:,jx], xstops, False) - 1
        j[j == -1] = 0; j[j == qdim] = qdim - 1
        py = np.zeros((size, qdim), 'd'); py[range(size), j] = 1

        if varall:
            print(np.einsum('ij,ik->ijk', px, py), xr)
            return xr, np.einsum('ij,ik->ijk', px, py), alphas
        else:
            return xr[:,[ix,jx]], np.einsum('ij,ik->ijk', px, py), alphas    
    else:
        ix = ['M','tc'].index(varx)

        i = np.digitize(xr[:,ix], xstops, False) - 1
        i[i == -1] = 0; i[i == qdim] = qdim - 1
        px = np.zeros((size, qdim), 'd'); px[range(size), i] = 1
  
        if varall:
            print(xr, px)
            return xr, px, alphas
        else:
            print(px)
            return xr[:,ix], px, alphas


def syntrainer(net, syntrain, lossfunction=None, iterations=300, 
               batchsize=None, initstep=1e-3, finalv=1e-5, clipgradient=None, validation=None,
               seed=None, single=True):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    """Trains network NN against training sets obtained from `syntrain`,
    iterating at most `iterations`; stops if the derivative of loss
    (averaged over 20 epochs) becomes less than `finalv`."""

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    indicatorloss = 'l' in lossfunction.__annotations__ and lossfunction.__annotations__['l'] == 'indicator'  

    if validation is not None:
        raise NotImplementedError

        vlabels = numpy2cuda(validation[1] if indicatorloss else validation[0], single)
        vinputs = numpy2cuda(validation[2], single)

    optimizer = optim.Adam(net.parameters(), lr=initstep)

    training_loss, validation_loss = [], []

    for epoch in range(iterations):
        correct = 0
        t0 = time.time()

        xtrue, indicator, inputs = syntrain()
        labels = numpy2cuda(indicator if indicatorloss else xtrue, single)

        if batchsize is None:
            batchsize = inputs.shape[0]
        batches = inputs.shape[0] // batchsize

        averaged_loss = 0.0    
        
        for i in range(batches):
        # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs[i*batchsize:(i+1)*batchsize])
            
            loss, dMc, dtc = lossfunction(outputs, labels[i*batchsize:(i+1)*batchsize])
            loss.backward()
      
            if clipgradient is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipgradient)
            ##Calculate standard deviation
           
            optimizer.step()
            if device=='cpu:0':
                correct += (torch.BoolTensor(abs(dMc) <= 0.2) & torch.BoolTensor(abs(dtc) <= 0.2)).count_nonzero().item()
            else:
                correct += (torch.cuda.BoolTensor(abs(dMc) <= 0.2) & torch.cuda.BoolTensor(abs(dtc) <= 0.2)).count_nonzero().item()

            ##Calculate accuracy of each epoch
            ##correct += (abs(dMc) <= 2*abs(sigma_Mc) and abs(dtc) <= 2*abs(sigma_tc)).nonzero()
            # print statistics
                        
            averaged_loss += loss.item()

        training_loss.append(averaged_loss/batches)
        
        accuracy = 100 * correct / batchsize
        
        if validation is not None:
            loss = lossfunction(net(vinputs), vlabels)
            validation_loss.append(loss.detach().cpu().item())

        if epoch == 1:
            print("One epoch = {:.1f} seconds.".format(time.time() - t0))

        if epoch % 10 == 0:
            print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch, iterations, training_loss[-1], accuracy))

        try:
            if len(training_loss) > iterations/10:
                training_rate = np.polyfit(range(20), training_loss[-20:], deg=1)[0]
                if training_rate < 0 and training_rate > -finalv:
                    print(f"Terminating at epoch {epoch} because training loss stopped improving sufficiently: rate = {training_rate}")
                    break

            if len(validation_loss) > iterations/10:
                validation_rate = np.polyfit(range(20), validation_loss[-20:], deg=1)[0]        
                if validation_rate > 0:
                    print(f"Terminating at epoch {epoch} because validation loss started worsening: rate = {validation_rate}")
                    break
        except:
            pass
          
    print("Final",training_loss[-1],validation_loss[-1] if validation is not None else '')

    if hasattr(net,'steps'):
        net.steps += iterations
    else:
        net.steps = iterations
    return training_loss
    
def sinc_f(x):
    ##Extract parameters for sinc function
    M = x[0]; tc = x[1]
    ##All Signals are cut-off to 200s long
    
    t_cutoff = 150

    N = 250

    z = np.linspace(-t_cutoff, t_cutoff, N)

    y=(z-tc) / M

    func = np.sinc(y)

    return func