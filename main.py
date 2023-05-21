import gwave
import pylab
import matplotlib.pyplot as plt

merge = "GW170608"
stain = "H1"
#for a given merge find and plot the GW

def detect_gw(merge, stain):
    merger, conditioner = gwave.precond_data(merge,stain,cond=True)
    template = gwave.model_wave(12,7,conditioner)
    psd = gwave.psd(conditioner)
    snr, time, snrp = gwave.peak_finder(template,conditioner, psd)
    wdata, aligned = gwave.alignment(conditioner,template,time,psd,snrp,merger)
    gwave.substract(conditioner,aligned,merger)
    return True

# Return the plot and data processed for given gravitational wave event and detector
detect_gw(merge, stain)



